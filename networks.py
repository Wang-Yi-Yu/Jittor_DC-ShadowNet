import jittor as jt
import jittor.nn as nn
from spectral_norm import spectral_norm


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=4, img_size=256, light=True):
        """
        Args:
            input_nc: the number of channels in input images
            output_nc: the number of channels in output images
            ngf: the number of generator filters in first conv layer
            n_blocks: the number of residual blocks in generator
            img_size: the size of input images
            light: if True, we use [b,256,1,1] in full connected layers; if False, we use [b,256,64,64] instead
        """
        assert n_blocks >= 0
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        # Initial convolution block
        DownBlock = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(),
        ]
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(),
            ]
        # Down-Sampling Bottleneck, Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU()
        # Gamma, Beta block
        FC = [
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.ReLU(),
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.ReLU(),
        ]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, "UpBlock1_" + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                ILN(int(ngf * mult / 2)),
                nn.ReLU(),
            ]
        # Output layer
        UpBlock2 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def execute(self, input):
        """
        Returns:
            out: the output of generator
            cam_logit:
            heatmap:
        """
        x = self.DownBlock(input)

        gap = nn.AdaptiveAvgPool2d(1)(x)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = nn.AdaptiveMaxPool2d(1)(x)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = jt.concat([gap_logit, gmp_logit], 1)
        x = jt.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = jt.sum(x, dim=1, keepdims=True)

        if self.light:
            x_ = nn.AdaptiveAvgPool2d(1)(x)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, "UpBlock1_" + str(i + 1))(x, gamma, beta)
        out = (self.UpBlock2(x) + input).tanh()

        return out, cam_logit, heatmap


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        """
        Args:
            dim: the number of channels in input and output features
            use_bias: if True, add a learnable bias to output
        """
        super().__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
        ]
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def execute(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def execute(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = jt.zeros((1, num_features, 1, 1))
        self.rho.fill_(0.9)

    def execute(self, input, gamma, beta):
        in_mean = jt.mean(input, dims=[2, 3], keepdims=True)
        in_var = jt.var(input, dims=[2, 3], keepdims=True)
        out_in = (input - in_mean) / jt.sqrt(in_var + self.eps)
        ln_mean = jt.mean(input, dims=[1, 2, 3], keepdims=True)
        ln_var = jt.var(input, dims=[1, 2, 3], keepdims=True)
        out_ln = (input - ln_mean) / jt.sqrt(ln_var + self.eps)
        out = (
                self.rho.expand(input.shape[0], -1, -1, -1) * out_in
                + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        )
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = jt.zeros((1, num_features, 1, 1))
        self.gamma = jt.zeros((1, num_features, 1, 1))
        self.beta = jt.zeros((1, num_features, 1, 1))
        self.gamma.fill_(1.0)

    def execute(self, input):
        in_mean = jt.mean(input, dims=[2, 3], keepdims=True)
        in_var = jt.var(input, dims=[2, 3], keepdims=True)
        out_in = (input - in_mean) / jt.sqrt(in_var + self.eps)
        ln_mean = jt.mean(input, dims=[1, 2, 3], keepdims=True)
        ln_var = jt.var(input, dims=[1, 2, 3], keepdims=True)
        out_ln = (input - ln_mean) / jt.sqrt(ln_var + self.eps)
        out = (
                self.rho.expand(input.shape[0], -1, -1, -1) * out_in
                + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        )
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, "rho"):
            w = module.rho.data
            w = w.clip(self.clip_min, self.clip_max)
            module.rho.data = w


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super().__init__()
        model = [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [
                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                nn.LeakyReLU(0.2),
            ]
        mult = 2 ** (n_layers - 2 - 1)  # 4
        model += [
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(0.2),
        ]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pad = nn.ReflectionPad2d(1)

        # FCN classification layer
        self.conv = spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*model)

    def execute(self, input):
        x = self.model(input)

        gap = nn.AdaptiveAvgPool2d(1)(x)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = nn.AdaptiveMaxPool2d(1)(x)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = jt.concat([gap_logit, gmp_logit], 1)
        x = jt.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        heatmap = jt.sum(x, dim=1, keepdims=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap