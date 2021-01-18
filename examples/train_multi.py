import numpy as np
from dreamz.cppn import get_xy_mesh, CPPNNet, UpsampleNet
from torch import optim
from dreamz.render import train_visualiser
from dreamz.torch_layers import Lambda
from torch import nn
from torchvision import datasets, models, transforms

device = 'cuda'

if 1:
    model = models.resnet18(pretrained=True).to(device)
    model = nn.Sequential(*(
        [i for i in model.children()][:-4] + [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Lambda(lambda x: x[:, :, 0, 0])]))
else:
    model = models.vgg11_bn(pretrained=True).to(device)
    model = nn.Sequential(*(
        [i for i in model.children()][:-1] + [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Lambda(lambda x: x[:, :, 0, 0])]))
model = model.eval()

from dreamz.googlenet import googlenet

def get_imagenet_model():
    model = googlenet(pretrained=True)
    return model.eval()

model = get_imagenet_model()
model = model.to(device)


import torch
mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

from dreamz.utils import display_tch_im

def get_cropped_mask(m, st_masks):
    out = []
    for st in st_masks:
        out.append(m.unsqueeze(0)[:, :, st:st + 228].cuda())
    out = torch.stack(out)
    return out

size = [59, 105]
xy = get_xy_mesh(size).to(device)

mask0 = torch.zeros([228, 406])
mask1 = torch.zeros([228, 406])
mask0[:, :406 // 2] = 1
mask1[:, 406 // 2:] = 1
masks = [mask0, mask1]
for m in masks:
    m.requires_grad = False
    m = m.to(device)
    
    
def train_cppn(cs, widths = [12] * 8):
    base = CPPNNet(widths, output_channels=widths[-1], bias=True)
    viz = UpsampleNet(base, reps=1).to(device)
    
    def imgnet_objective(output, st_masks, masks=masks, cs=cs):
        loss = None
        this_masks = [get_cropped_mask(m, st_masks) for m in masks]
        for m, c in zip(this_masks, cs):
            to_model = ((output - mean) / std)
            to_model *= m
            r = model(to_model)
            this_loss = -r[:, c].mean()
            if loss is None:
                loss = this_loss
            else:
                loss += this_loss
    #     return torch.mean((r - targ) ** 2)
        return loss
        
    def im_gen_fn(pct_done=0.0, num=10):
        xy_crop = []
        st_masks = []
        for i in range(num):
            x0 = np.random.randint(0, 105 - 59)
            st_masks.append(int(x0 * 228 / 59))
            xy_crop.append(xy[:, :, :, x0:x0 + 59])
        xy_crop = torch.cat(xy_crop, 0)
        return viz(xy_crop), st_masks
    
    opt = optim.Adam(viz.parameters(), lr=0.002)

    train_visualiser(imgnet_objective, im_gen_fn, opt, iters=150, log_interval=0)
    
    return viz


import os
os.makedirs("/root/cppns/", exist_ok=True)
for i in range(10):
    for c0 in range(512):
        for c1 in range(512):
            print(i, c0, c1)
            viz = train_cppn([c0, c1])
            torch.save(viz.state_dict(), f"/root/cppns/{i:03d}-{c0:03d}-{c1:03d}.pth")
