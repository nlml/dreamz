import os
import numpy as np
import torch
import torch.nn as nn
from dreamz.cppn import CPPNNet, UpsampleNet


class Wrapper(nn.Module):
    def __init__(self, m):
        super(Wrapper, self).__init__()
        self.m = m

    def forward(self, x, o):
        o = o.view([1, 2, 1, 1])
        o = o.mean(1, keepdim=True)
        o = o.repeat([x.size(0), 1, x.size(2), x.size(3)])
        x = torch.cat([x, o], 1)
        x = self.m(x)
        return x.permute(0, 2, 3, 1)


def get_net(device):
    widths = [24] * 8
    basenet = CPPNNet(widths, input_channels=3)
    viz = UpsampleNet(basenet, reps=1)
    base = "/home/liam/dreamz/data/state_dicts_v2/"
    state_dicts = [torch.load(base + i) for i in os.listdir(base)]
    print(os.listdir(base), "sds")
    state_dicts += state_dicts
    state_dicts += state_dicts
    print(len(state_dicts))
    np.random.shuffle(state_dicts)
    viz.load_state_dict(state_dicts[0])
    m = Wrapper(viz).to(device)
    return state_dicts, viz, m
