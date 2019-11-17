import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dreamz.torch_layers import Lambda


def get_xy_mesh(size=224, r=3.0**0.5):
    if type(size) is int:
        size = [size, size]
    if size[0] >= size[1]:
        ratio = size[0] / size[1]
        rng0 = torch.linspace(-r * ratio, r * ratio, size[0])
        rng1 = torch.linspace(-r, r, size[1])
    else:
        ratio = size[1] / size[0]
        rng0 = torch.linspace(-r, r, size[0])
        rng1 = torch.linspace(-r * ratio, r * ratio, size[1])
    xy = torch.stack([*torch.meshgrid(rng0, rng1)], -1).float()  # noqa
    xy = torch.unsqueeze(xy, 0).permute(0, 3, 1, 2)
    return xy


def composite_activation(x):
    x = torch.atan(x)
    return torch.cat([x / 0.67, (x * x) / 0.6], 1)


def composite_activation_unbiased(x):
    x = torch.atan(x)
    return torch.cat([x / 0.6706, (x * x - 0.45) / 0.396], 1)


def get_cppn_im_gen_fn_and_opt(size, device, widths=None, opt=None):
    xy = get_xy_mesh(size).to(device)
    if widths is None:
        widths = [24] * 3
    viz = CPPNNet(widths).to(device)
    if opt is None:
        opt = lambda p: optim.SGD(p, lr=0.1, momentum=0.9)  # noqa
    return lambda _: (viz(xy), None), opt(viz.parameters())


class CPPNNet(nn.Module):
    def __init__(
            self,
            n_channels_list,
            kernel_size=1,
            act_fn=composite_activation,
            use_bn=False,
            input_channels=2,
            output_channels=3):

        super(CPPNNet, self).__init__()
        self.use_bn = use_bn
        self.layers = []
        self.input_channels = input_channels
        self.output_channels = output_channels
        n_channels_list = [input_channels] + n_channels_list + [output_channels]
        chans0 = n_channels_list[0]
        for i, chans1 in enumerate(n_channels_list[1:]):
            this = nn.Sequential()
            if self.use_bn:
                this.add_module('bn{}'.format(i), nn.BatchNorm2d(chans0))
            this.add_module('conv{}'.format(i), nn.Conv2d(chans0, chans1, kernel_size))
            # Initialise the weight to preserve mean/std
            nn.init.normal_(
                this[-1].weight,
                std=np.sqrt(1 / (chans0 * (kernel_size ** 2)))
            )
            if i < len(n_channels_list) - 2:
                this.add_module('act{}'.format(i), Lambda(act_fn))
                chans0 = chans1 * 2
            self.layers.append(this)
        self.layers = nn.Sequential(*self.layers)
        self.final_act = nn.Sigmoid()

    def do_debug_prints(self):
        print('weight', self.layers[-2][0].weight.mean().item())
        print('grad', torch.abs(self.layers[-2][0].weight.grad).mean())

    def forward(self, x, debug=False):
        if debug:
            for fc in self.layers:
                x = fc(x)
                print(x.view(-1).mean(), x.view(-1).std())
        else:
            x = self.layers(x)
        x = self.final_act(x)
        return x


class UpsampleNet(nn.Module):
    def __init__(self, base_model, reps=2, output_channels=3,
                 interp_mode='bilinear'):
        super(UpsampleNet, self).__init__()
        self.base_model = base_model
        self.interp_mode = interp_mode
        self.reps = reps
        self.output_channels = output_channels
        
        s = self.base_model.output_channels
        
        self.conv1 = self.get_group_of_layers(self.reps, s, s)
        self.conv2 = self.get_group_of_layers(self.reps - 1, s * 2, s)
        self.conv3 = nn.Conv2d(s * 2, output_channels, 3)
        self.final_act = nn.Sigmoid()
        
    def get_group_of_layers(self, reps, s0, s, k=1):
        this = []
        for i in range(reps):
            this += [nn.Conv2d(s0, s, k)]
            nn.init.normal_(
                this[-1].weight,
                std=np.sqrt(1 / (s0 * (k ** 2)))
            )
            this += [Lambda(composite_activation)]
            s0 = s * 2
        return nn.Sequential(*this)

    def forward(self, x):
        x = self.base_model.layers(x)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode=self.interp_mode)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode=self.interp_mode)
        x = self.conv3(x)
        x = self.final_act(x)
        return x[:, :, 3:-3, 3:-3]


class UpsampleNetNew(nn.Module):
    def __init__(self, input_channels, reps=2, output_channels=3,
                 interp_mode='bilinear', extra_upscale=False, k=7, s=10):
        super(UpsampleNetNew, self).__init__()
        self.interp_mode = interp_mode
        self.reps = reps
        self.output_channels = output_channels
        self.extra_upscale = extra_upscale

        self.conv1 = self.get_layer(input_channels, s, k)
        self.conv2 = self.get_layer(s * 2, s, k)
        inps = s * 2
        self.conv3 = nn.Conv2d(inps, output_channels, 3, bias=False)
        nn.init.normal_(self.conv3.weight, std=np.sqrt(1 / (inps * (k ** 2))))
        self.final_act = nn.Sigmoid()
        
    def get_layer(self, s0, s, k=1):
        this = nn.Conv2d(s0, s, k, bias=False, dilation=2)
        nn.init.normal_(this.weight, std=np.sqrt(1 / (s0 * (k ** 2))))
        this = [this, Lambda(composite_activation)]
        return nn.Sequential(*this)

    def forward(self, x):
        if self.extra_upscale:
            x = F.interpolate(x, scale_factor=2, mode=self.interp_mode)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=3, mode=self.interp_mode)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=3, mode=self.interp_mode)
        x = self.conv3(x)
        x = self.final_act(x)
        return x[:, :, 3:-3, 3:-3]
