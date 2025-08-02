import os, cv2
import jittor as jt
import jittor.nn as nn
from jittor import models as visionmodels


def check_folder(log_dir):
    """
    If there are any files named `log_dir`, it will be created.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in "true"


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)  # [c,h,w]->[h,w,c]


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def softmask_generator(shadow, shadow_free):
    fake_A2B_ = (shadow_free + 1.0) / 2.0
    real_A_ = (shadow + 1.0) / 2.0
    diffrAfB = jt.mean((fake_A2B_ - real_A_), dim=1, keepdims=True)
    diffrAfB[diffrAfB < 0.05] = 0
    mask1crAfB = (diffrAfB - jt.min(diffrAfB)) / (jt.max(diffrAfB) - jt.min(diffrAfB))
    mask1crAfB = mask1crAfB * 2 - 1
    softmask = jt.concat((mask1crAfB, mask1crAfB, mask1crAfB), dim=1)
    return softmask


def smooth_loss_masked(pred_map, mask):
    def gradient(pred, mask):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        mask_D_dy = mask[:, :, 1:] - mask[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        mask_D_dx = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        return D_dx, D_dy, mask_D_dx, mask_D_dy

    dx, dy, mask_dx, mask_dy = gradient(pred_map, mask)
    loss = (mask_dx * dx).abs().mean() + (mask_dy * dy).abs().mean()
    return loss


class PerceptualLossVgg16(nn.Module):
    def __init__(self, weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16, self).__init__()
        self.vgg = Vgg16().cuda()
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.indices = indices or [3, 8, 15, 22]
        if normalize:
            self.normalize = MeanShift(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True
            ).cuda()
        else:
            self.normalize = None

    def execute(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22]
        out = []
        for i in range(indices[-1] + 1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = jt.Var(data_std)
        self.weight = jt.reshape(jt.init.eye(c), (c, c, 1, 1))
        if norm:
            self.weight /= jt.reshape(std, (c, 1, 1, 1))
            self.bias = -1 * data_range * jt.Var(data_mean)
            self.bias /= std
        else:
            self.weight *= jt.reshape(std, (c, 1, 1, 1))
            self.bias = data_range * jt.Var(data_mean)
        self.requires_grad = False
