import time
from skimage.io import imsave
from dreamz.utils import get_latest_filename, tch_im_to_np
from dreamz.cppn import get_xy_mesh, CPPNNet
from dreamz.render import train_visualiser
from dreamz.torch_utils import Lambda, adjust_learning_rate
from dreamz.cppn import composite_activation
import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
from torch import optim
import numpy as np


class Net(nn.Module):
    def __init__(self, base_model, reps=2, output_channels=3):
        super(Net, self).__init__()
        self.base_model = base_model

        s = self.base_model.output_channels

        self.conv1 = self.get_group_of_layers(reps, s, s)
        self.conv2 = self.get_group_of_layers(reps - 1, s * 2, s)
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
#             this += [nn.ReLU6()]
            this += [Lambda(composite_activation)]
            s0 = s * 2
        return nn.Sequential(*this)

    def forward(self, x):
        x = self.base_model.layers(x)
        x = self.conv1(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.conv2(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.conv3(x)
        x = self.final_act(x)
        return x[:, :, 3:-3, 3:-3]


def train(size, widths, imagenet_model, chan_to_opt):
    base = CPPNNet(widths, output_channels=widths[-1])
    viz = Net(base, reps=1).to(device)
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

    def imgnet_objective(output):
        r = imagenet_model((output - mean) / std)
    #     return torch.mean((r - targ) ** 2)
        return -r[:, chan_to_opt].mean()

    xy = get_xy_mesh(size).to(device)

    def im_gen_fn(num=16):
        xy_crop = []
        for i in range(num):
            x0 = np.random.randint(0, 105 - 59)
            xy_crop.append(xy[:, :, :, x0:x0 + 59])
        xy_crop = torch.cat(xy_crop, 0)
        return viz(xy_crop)
    opt = optim.Adam(viz.parameters(), lr=0.002)
    train_visualiser(imgnet_objective, im_gen_fn, opt, iters=150, log_interval=0)
    adjust_learning_rate(opt, 0.1)
    train_visualiser(imgnet_objective, im_gen_fn, opt, iters=50, log_interval=0)
    return viz


def get_imagenet_model():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*(
        [i for i in model.children()][:-2] + [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Lambda(lambda x: x[:, :, 0, 0])]))
    # model = nn.Sequential(*(
    #     [i for i in model.children()][:-3] + [
    #         nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    #         Lambda(lambda x: x[:, :, 0, 0])]))
    return model.eval()


device = 'cuda'
ims_savedir = '../../data/output_ims2/'
widths = [20] * 8
size = [59, 105]

imagenet_model = get_imagenet_model().to(device)


def train_and_plot(chan_to_opt):
    now = time.time()
    print('Training {}'.format(chan_to_opt))
    viz = train(size, widths, imagenet_model, chan_to_opt)
    print('Took {} seconds'.format(time.time() - now))
    now = time.time()
    print('Saving {}'.format(chan_to_opt))
    xy_big = get_xy_mesh([277, 502]).to(device)
    res = viz(xy_big)
    imsave(get_latest_filename(ims_savedir), tch_im_to_np(res))
    print('Took {} seconds'.format(time.time() - now))


for chan_to_opt in range(5000):
    train_and_plot(chan_to_opt)
