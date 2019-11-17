import os
import numpy as np
import torch
import torch.nn as nn
from dreamz.cppn import CPPNNet, UpsampleNetNew


class NetWrap(nn.Module):
    def __init__(self, base_model, upsample_model):
        super(NetWrap, self).__init__()
        self.base_model = base_model
        self.upsample_model = upsample_model
    def forward(self, x):
        return self.upsample_model(self.base_model(x))


class Wrapper(nn.Module):
    def __init__(self, m):
        super(Wrapper, self).__init__()
        self.m = m

    def forward(self, x, o):
        o = o.view([1, 6, 1, 1])
        # o = o.mean(1, keepdim=True)
        o = o.repeat([x.size(0), 1, x.shape[2], x.shape[3]])
        x = torch.cat([x, o], 1)
        x = self.m(x)
        return x.permute(0, 2, 3, 1)


def get_net(device):
    widths = [24] * 10
    NOISE_VEC_DIM = 6
    base = CPPNNet(widths, input_channels=2 + NOISE_VEC_DIM)
    upsampler = UpsampleNetNew(base.output_channels, reps=1, extra_upscale=False, k=5)
    viz = NetWrap(base, upsampler)
    base = '/home/liam/dreamz/data/state_dicts_v3/'
    state_dicts = [torch.load(base + i) for i in os.listdir(base)]
    print(os.listdir(base), 'sds')
    state_dicts += state_dicts
    state_dicts += state_dicts
    print(len(state_dicts))
    np.random.shuffle(state_dicts)
    viz.load_state_dict(state_dicts[0])
    m = Wrapper(viz).to(device)
    return state_dicts, viz, m
