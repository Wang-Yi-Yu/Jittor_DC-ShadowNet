from dataset import ImageFolder
from jittor import transform as transforms
from jittor.dataset import DataLoader
from networks import *
from utils_loss import *
from glob import glob
from PIL import Image
import numpy as np


class DCShadowNet(object):
    def __init__(self, args):
        self.model_name = "DCShadowNet"
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datasetpath = args.datasetpath
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.dom_weight = args.dom_weight
        self.use_ch_loss = args.use_ch_loss
        self.use_pecp_loss = args.use_pecp_loss
        self.use_smooth_loss = args.use_smooth_loss
        if args.use_ch_loss == True:
            self.ch_weight = args.ch_weight
        if args.use_pecp_loss == True:
            self.pecp_weight = args.pecp_weight
        if args.use_smooth_loss == True:
            self.smooth_weight = args.smooth_weight
        self.n_res = args.n_res
        self.n_dis = args.n_dis
        self.img_size = args.img_size
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch
        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.use_original_name = args.use_original_name
        self.im_suf_A = args.im_suf_A
        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset)

    def build_model(self):
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.testA = ImageFolder(os.path.join("dataset", self.datasetpath, "testA"), self.test_transform)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)

        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True)

    def load(self, dir, step):
        params = jt.load(os.path.join(dir, self.dataset + "_params_%07d.pkl" % step))
        self.genA2B.load_state_dict(params["genA2B"])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, "model", "*.pkl"))
        if not len(model_list) == 0:
            model_list.sort()
            print("model_list", model_list)
            iter = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load(os.path.join(self.result_dir, self.dataset, "model"), iter)
            print(" [*] Load SUCCESS")
            self.genA2B.eval()

            path_fakeB = os.path.join(self.result_dir, self.dataset, str(iter) + "/outputB")
            if not os.path.exists(path_fakeB):
                os.makedirs(path_fakeB)
            path_realAfakeB = os.path.join( self.result_dir, self.dataset, str(iter) + "/inputA_outputB")

            if not os.path.exists(path_realAfakeB):
                os.makedirs(path_realAfakeB)

            if self.use_original_name:
                self.test_list = [
                    os.path.splitext(f)[0]
                    for f in os.listdir(os.path.join("dataset", self.datasetpath, "testA"))
                    if f.endswith(self.im_suf_A)
                ]

                for n, img_name in enumerate(self.test_list):
                    print("predicting: %d / %d" % (n + 1, len(self.test_list)))

                    img = Image.open(os.path.join("dataset", self.datasetpath, "testA", img_name + self.im_suf_A)).convert("RGB")
                    img = jt.array(self.test_transform(img))
                    real_A = img.unsqueeze(0)
                    fake_A2B, _, _ = self.genA2B(real_A)
                    A_real = RGB2BGR(tensor2numpy(denorm(real_A[0])))
                    B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

                    A2B = np.concatenate((A_real, B_fake), 1)
                    cv2.imwrite(os.path.join(path_fakeB, "%s.png" % img_name), B_fake * 255.0)
                    cv2.imwrite(os.path.join(path_realAfakeB, "%s.png" % img_name), A2B * 255.0)

            else:
                for n, (real_A, _) in enumerate(self.testA_loader):
                    print("predicting: %d / %d" % (n + 1, len(self.testA_loader)))

                    real_A = real_A
                    fake_A2B, _, _ = self.genA2B(real_A)
                    A_real = RGB2BGR(tensor2numpy(denorm(real_A[0])))
                    B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

                    A2B = np.concatenate((A_real, B_fake), 1)
                    cv2.imwrite(os.path.join(path_fakeB, "%d.png" % (n + 1)), B_fake * 255.0)
                    cv2.imwrite(os.path.join(path_realAfakeB, "%d.png" % (n + 1)), A2B * 255.0)