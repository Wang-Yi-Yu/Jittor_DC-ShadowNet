import jittor as jt
import jittor.transform as transforms
from jittor import nn
import os, time
from dataset import default_loader as img_loader


def calc_rmse(fake, real):
    mse = nn.mse_loss(fake, real)
    rmse = jt.sqrt(mse)
    return rmse.item()


def calc_psnr(fake, real):
    mse = nn.mse_loss(fake, real)
    psnr = 10.0 * jt.log(255.0**2 / mse) / jt.log(10)
    return psnr.item()


jt.flags.use_cuda = 1
fake_dir = "evaluation/fake_shadow-free"
real_dir = "evaluation/real_shadow-free"
fake_files = sorted(os.listdir(fake_dir))
real_files = sorted(os.listdir(real_dir))
assert len(fake_files) == len(real_files)
print(" [*] Load SUCCESS")
start = time.time()
sum_rmse, min_rmse, max_rmse, min_rmse_id, max_rmse_id = 0, float("inf"), 0, 0, 0
sum_psnr, min_psnr, max_psnr, min_psnr_id, max_psnr_id = 0, float("inf"), 0, 0, 0
for i, fake_file in enumerate(fake_files):
    real_file = real_files[i]
    fake_img = img_loader(os.path.join(fake_dir, fake_file))
    real_img = img_loader(os.path.join(real_dir, real_file))
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    fake_tensor = jt.float32(transform(fake_img)) * 255
    real_tensor = jt.float32(transform(real_img)) * 255
    rmse = calc_rmse(fake_tensor, real_tensor)
    psnr = calc_psnr(fake_tensor, real_tensor)
    print("pic %4d : RMSE = %9.6f , PSNR = %9.6f" % (i + 1, rmse, psnr))
    sum_rmse += rmse
    min_rmse, min_rmse_id = (
        (rmse, i + 1) if rmse < min_rmse else (min_rmse, min_rmse_id)
    )
    max_rmse, max_rmse_id = (
        (rmse, i + 1) if rmse > max_rmse else (max_rmse, max_rmse_id)
    )
    sum_psnr += psnr
    min_psnr, min_psnr_id = (
        (psnr, i + 1) if psnr < min_psnr else (min_psnr, min_psnr_id)
    )
    max_psnr, max_psnr_id = (
        (psnr, i + 1) if psnr > max_psnr else (max_psnr, max_psnr_id)
    )
print(" [*] Evaluate COMPLETE in %.3fs" % (time.time() - start))
print(
    "RMSE: average = %9.6f , min = %9.6f(%d) , max = %9.6f(%d)"
    % (sum_rmse / len(fake_files), min_rmse, min_rmse_id, max_rmse, max_rmse_id)
)
print(
    "PSNR: average = %9.6f , min = %9.6f(%d) , max = %9.6f(%d)"
    % (sum_psnr / len(fake_files), min_psnr, min_psnr_id, max_psnr, max_psnr_id)
)
