import time
from dataset import ImageFolder
from jittor import transform as transforms
from jittor.dataset import DataLoader
from networks import *
from utils_loss import *
from glob import glob
import matplotlib.pyplot as plt
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
        self.img_ch = args.img_ch
        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset)

    def build_model(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.trainA = ImageFolder(os.path.join("dataset", self.dataset, "trainA"), train_transform)  # shadow
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB = ImageFolder(os.path.join("dataset", self.dataset, "trainB"), train_transform)  # shadow-free
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        if self.use_ch_loss:
            self.trainC = ImageFolder(os.path.join("dataset", self.dataset, "trainC"), train_transform)  # offline load physics ch_norm
            self.trainC_loader = DataLoader(self.trainC, batch_size=self.batch_size, shuffle=False)

        self.testA = ImageFolder(os.path.join("dataset", self.dataset, "testA"), test_transform)  # shadow
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB = ImageFolder(os.path.join("dataset", self.dataset, "testB"), test_transform)  # shadow-free
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        if self.use_ch_loss:
            self.testC = ImageFolder(os.path.join("dataset", self.dataset, "testC"), test_transform)   # offline load physics ch_norm
            self.testC_loader = DataLoader(self.testC, batch_size=1, shuffle=False)

        """ 
        A refers shadow, B refers shadow-free
        G refers global, which means a discriminator with a depth of 7
        L refers local, which means a discriminator with a depth of 5
        """

        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()

        self.G_optim = jt.optim.Adam(
            self.genA2B.parameters() + self.genB2A.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )
        self.D_optim = jt.optim.Adam(
            self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )

        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        start_iter = 1  # iteration in real=self.iteration-start_iter

        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, "model", "*.pkl"))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load(os.path.join(self.result_dir, self.dataset, "model"), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.lr = max(0, self.G_optim.lr - (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2))
                    self.D_optim.lr = max(0, self.D_optim.lr - (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2))

        G_losses, D_losses = [], []
        print("training start !")
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.lr = max(0, self.G_optim.lr - (self.lr / (self.iteration // 2)))
                self.D_optim.lr = max(0, self.D_optim.lr - (self.lr / (self.iteration // 2)))

            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)
            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)
            if self.use_ch_loss:
                try:
                    real_C, _ = next(trainC_iter)
                except:
                    trainC_iter = iter(self.trainC_loader)
                    real_C, _ = next(trainC_iter)

            # Update D
            fake_A2B, _, _ = self.genA2B(real_A)  # shadow      --genA2B-> fake_A2B
            fake_B2A, _, _ = self.genB2A(real_B)  # shadow-free --genB2A-> fake_B2A

            real_GA_logit, real_GA_Dom_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_Dom_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_Dom_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_Dom_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_Dom_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_Dom_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = (self.MSE_loss(real_GA_logit, jt.ones_like(real_GA_logit))
                            + self.MSE_loss(fake_GA_logit, jt.zeros_like(fake_GA_logit)))
            D_ad_Dom_loss_GA = (self.MSE_loss(real_GA_Dom_logit, jt.ones_like(real_GA_Dom_logit))
                                + self.MSE_loss(fake_GA_Dom_logit, jt.zeros_like(fake_GA_Dom_logit)))
            D_ad_loss_LA = (self.MSE_loss(real_LA_logit, jt.ones_like(real_LA_logit))
                            + self.MSE_loss(fake_LA_logit, jt.zeros_like(fake_LA_logit)))
            D_ad_Dom_loss_LA = (self.MSE_loss(real_LA_Dom_logit, jt.ones_like(real_LA_Dom_logit))
                                + self.MSE_loss(fake_LA_Dom_logit, jt.zeros_like(fake_LA_Dom_logit)))

            D_ad_loss_GB = (self.MSE_loss(real_GB_logit, jt.ones_like(real_GB_logit))
                            + self.MSE_loss(fake_GB_logit, jt.zeros_like(fake_GB_logit)))
            D_ad_Dom_loss_GB = (self.MSE_loss(real_GB_Dom_logit, jt.ones_like(real_GB_Dom_logit))
                                + self.MSE_loss(fake_GB_Dom_logit, jt.zeros_like(fake_GB_Dom_logit)))
            D_ad_loss_LB = (self.MSE_loss(real_LB_logit, jt.ones_like(real_LB_logit))
                            + self.MSE_loss(fake_LB_logit, jt.zeros_like(fake_LB_logit)))
            D_ad_Dom_loss_LB = (self.MSE_loss(real_LB_Dom_logit, jt.ones_like(real_LB_Dom_logit))
                                + self.MSE_loss(fake_LB_Dom_logit, jt.zeros_like(fake_LB_Dom_logit)))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_Dom_loss_GA + D_ad_loss_LA + D_ad_Dom_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_Dom_loss_GB + D_ad_loss_LB + D_ad_Dom_loss_LB)
            Discriminator_loss = D_loss_A + D_loss_B
            self.D_optim.step(Discriminator_loss)

            # Update G
            fake_A2B, fake_A2B_Dom_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_Dom_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_Dom_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_Dom_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_Dom_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_Dom_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, jt.ones_like(fake_GA_logit))
            G_ad_Dom_loss_GA = self.MSE_loss(fake_GA_Dom_logit, jt.ones_like(fake_GA_Dom_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, jt.ones_like(fake_LA_logit))
            G_ad_Dom_loss_LA = self.MSE_loss(fake_LA_Dom_logit, jt.ones_like(fake_LA_Dom_logit))

            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, jt.ones_like(fake_GB_logit))
            G_ad_Dom_loss_GB = self.MSE_loss(fake_GB_Dom_logit, jt.ones_like(fake_GB_Dom_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, jt.ones_like(fake_LB_logit))
            G_ad_Dom_loss_LB = self.MSE_loss(fake_LB_Dom_logit, jt.ones_like(fake_LB_Dom_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_dom_loss_A = (self.BCE_loss(fake_B2A_Dom_logit, jt.ones_like(fake_B2A_Dom_logit))
                            + self.BCE_loss(fake_A2A_Dom_logit, jt.zeros_like(fake_A2A_Dom_logit)))
            G_dom_loss_B = (self.BCE_loss(fake_A2B_Dom_logit, jt.ones_like(fake_A2B_Dom_logit))
                            + self.BCE_loss(fake_B2B_Dom_logit, jt.zeros_like(fake_B2B_Dom_logit)))

            if self.use_pecp_loss:
                selfpecpvgg_loss = PerceptualLossVgg16(weights=[1.0], indices=[22])
                loss_selfpecp = selfpecpvgg_loss(fake_A2B, real_A)

            if self.use_smooth_loss:
                gen_mask = softmask_generator(real_A, fake_A2B)
                loss_smooth = smooth_loss_masked(fake_A2B, gen_mask)

            if self.use_ch_loss:
                fake_A2B_ = (fake_A2B + 1.0) / 2.0
                ch_z = fake_A2B_ / fake_A2B_.sum(dim=1, keepdims=True).clamp(1e-8)
                ch_z = 2 * ch_z - 1
                ch_norm = real_C
                loss_ch = self.L1_loss(ch_z, ch_norm)

            G_loss_A = (
                self.adv_weight * (G_ad_loss_GA + G_ad_Dom_loss_GA + G_ad_loss_LA + G_ad_Dom_loss_LA)
                + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A
                + self.dom_weight * G_dom_loss_A
            )
            G_loss_B = (
                self.adv_weight * (G_ad_loss_GB + G_ad_Dom_loss_GB + G_ad_loss_LB + G_ad_Dom_loss_LB)
                + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B
                + self.dom_weight * G_dom_loss_B
            )
            Generator_loss = G_loss_A + G_loss_B

            if self.use_ch_loss == True:
                Generator_loss = Generator_loss + loss_ch
            if self.use_pecp_loss == True:
                Generator_loss = Generator_loss + loss_selfpecp
            if self.use_smooth_loss == True:
                Generator_loss = Generator_loss + loss_smooth

            self.G_optim.step(Generator_loss)

            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print(
                "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f"
                % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)
            )

            with jt.no_grad():
                if step == start_iter:
                    G_losses.append(Generator_loss.item())
                    D_losses.append(Discriminator_loss.item())
                if step % self.print_freq == 0:
                    G_losses.append(Generator_loss.item())
                    D_losses.append(Discriminator_loss.item())
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 4, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = next(trainA_iter)
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = next(trainA_iter)
                        try:
                            real_B, _ = next(trainB_iter)
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = next(trainB_iter)

                        fake_A2B, _, _ = self.genA2B(real_A)
                        fake_B2A, _, _ = self.genB2A(real_B)
                        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                        fake_B2A2B, _, _ = self.genA2B(fake_B2A)
                        fake_A2A, _, _ = self.genB2A(real_A)
                        fake_B2B, _, _ = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((
                            RGB2BGR(tensor2numpy(denorm(real_A[0]))), RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                            RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))), RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0]))),
                        ), 0)), 1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = next(testA_iter)
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = next(testA_iter)
                        try:
                            real_B, _ = next(testB_iter)
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = next(testB_iter)
                        if self.use_ch_loss:
                            try:
                                real_C_test, _ = next(testC_iter)
                            except:
                                testC_iter = iter(self.testC_loader)
                                real_C_test, _ = next(testC_iter)

                        fake_A2B, _, _ = self.genA2B(real_A)
                        fake_B2A, _, _ = self.genB2A(real_B)
                        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                        fake_B2A2B, _, _ = self.genA2B(fake_B2A)
                        fake_A2A, _, _ = self.genB2A(real_A)
                        fake_B2B, _, _ = self.genA2B(real_B)

                        if self.use_smooth_loss == True:
                            gen_mask = softmask_generator(real_A, fake_A2B)
                            A2B = np.concatenate((A2B, np.concatenate((
                                RGB2BGR(tensor2numpy(denorm(real_A[0]))), RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                RGB2BGR(tensor2numpy(denorm(gen_mask[0]))), RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0]))),
                            ), 0)), 1)

                        if self.use_ch_loss == True:
                            fake_A2B_ = (fake_A2B + 1.0) / 2.0
                            ch_z = fake_A2B_ / fake_A2B_.sum(dim=1, keepdims=True).clamp(1e-8)
                            ch_z_test = 2 * ch_z - 1
                            ch_norm_test = real_C_test
                            A2B = np.concatenate((A2B, np.concatenate((
                                RGB2BGR(tensor2numpy(denorm(real_A[0]))), RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                 RGB2BGR(tensor2numpy(denorm(ch_norm_test[0]))), RGB2BGR(tensor2numpy(denorm(ch_z_test[0]))),
                            ), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, "train_img", "A2B_%07d.png" % step), A2B * 255.0)

                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, "model"), step)

                if step % 1000 == 0:
                    params = {}
                    params["genA2B"] = self.genA2B.state_dict()
                    params["genB2A"] = self.genB2A.state_dict()
                    params["disGA"] = self.disGA.state_dict()
                    params["disGB"] = self.disGB.state_dict()
                    params["disLA"] = self.disLA.state_dict()
                    params["disLB"] = self.disLB.state_dict()
                    jt.save(params, os.path.join(self.result_dir, self.dataset + "_params_latest.pkl"))

        plt.rcParams.update(
            {
                "font.size": 28,
                "axes.titlesize": 32,
                "axes.labelsize": 30,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "legend.fontsize": 26,
            }
        )
        plt.figure(figsize=(50, 10))

        real_iteration = range(0, len(G_losses) * self.print_freq, self.print_freq)
        plt.plot(
            real_iteration,
            G_losses,
            color="b",
            alpha=0.75,
            linestyle="--",
            linewidth=2,
        )

        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("The Generator Loss during training")
        plt.savefig("Generator loss.png", dpi=300)

        plt.rcParams.update(
            {
                "font.size": 28,
                "axes.titlesize": 32,
                "axes.labelsize": 30,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "legend.fontsize": 26,
            }
        )
        plt.figure(figsize=(50, 10))

        plt.plot(
            real_iteration,
            D_losses,
            color="r",
            alpha=0.75,
            linestyle="--",
            linewidth=2,
        )
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("The Discriminator Loss during training")
        plt.savefig("Discriminator loss.png", dpi=300)

    def save(self, dir, step):
        params = {}
        params["genA2B"] = self.genA2B.state_dict()
        params["genB2A"] = self.genB2A.state_dict()
        params["disGA"] = self.disGA.state_dict()
        params["disGB"] = self.disGB.state_dict()
        params["disLA"] = self.disLA.state_dict()
        params["disLB"] = self.disLB.state_dict()
        jt.save(params, os.path.join(dir, self.dataset + "_params_%07d.pkl" % step))

    def load(self, dir, step):
        params = jt.load(os.path.join(dir, self.dataset + "_params_%07d.pkl" % step))
        self.genA2B.load_state_dict(params["genA2B"])
        self.genB2A.load_state_dict(params["genB2A"])
        self.disGA.load_state_dict(params["disGA"])
        self.disGB.load_state_dict(params["disGB"])
        self.disLA.load_state_dict(params["disLA"])
        self.disLB.load_state_dict(params["disLB"])
