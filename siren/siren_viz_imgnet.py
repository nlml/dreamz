from tqdm import tqdm
from dreamz.googlenet import googlenet
import subprocess
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
import PIL
import matplotlib.pyplot as plt
import os
import random
import imageio
import glob
from siren import Siren, get_mgrid


device = "cuda"


im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape


def displ(img, pre_scaled=True):
    img = np.array(img)[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48 * 4, 32 * 4)
    imageio.imwrite(str(3) + ".png", np.array(img))
    return display.Image(str(3) + ".png")


def card_padded(im, to_pad=3):
    return np.pad(
        np.pad(
            np.pad(im, [[1, 1], [1, 1], [0, 0]], constant_values=0),
            [[2, 2], [2, 2], [0, 0]],
            constant_values=1,
        ),
        [[to_pad, to_pad], [to_pad, to_pad], [0, 0]],
        constant_values=0,
    )


def get_imagenet_model():
    model = googlenet(pretrained=True)
    return model.eval()


def forward(self, coords, act_fn=None, dim=2):
    h, w = coords.shape[:2]
    coords = coords.reshape(-1, dim)
    s = int(coords.shape[0] ** 0.5)
    if act_fn is not None:
        self.net._periodic_activation = act_fn
    output = self.net(coords)
    return output.view(1, h, w, 3).permute(0, 3, 1, 2)


# Periodic activation functions
x = torch.linspace(-2 * np.pi, 2 * np.pi, 100)
triangle = lambda x: (2 / np.pi) * torch.arcsin(torch.sin(x))
funky = lambda x: (
    torch.relu(torch.sin(x)) - torch.relu(torch.sin(x * 0.5 + np.pi * 0.5))
)
funky = lambda x: torch.sin(x)
funky = lambda x: (-2 / np.pi) * torch.arctan(
    1 / torch.tan(x / 2 + np.pi * 0.5)
)
funky = lambda x: torch.arccos(torch.cos(x + np.pi * 0.5)) / np.pi * 2 - 1
funky = lambda x: torch.relu(torch.cos(x)) - torch.relu(
    torch.cos(x + np.pi * 0.5)
)
step = 0
funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
    torch.cos(x + step + np.pi * 0.5)
)
# funky = triangle

plt.plot(x, torch.sin(x), label="sin")
# plt.plot(x, torch.cos(x), label='cos')
# plt.plot(x, triangle(x), label='triangle')
plt.plot(x, funky(x), label="funky")
plt.legend()
plt.show()


def checkin(model, loss):
    print(loss)
    # with torch.no_grad():
    #     al = nom(model(get_mgrid(sideX)).cpu()).numpy()
    # for allls in al:
    #     displ(allls)
    #     display.display(display.Image(str(3) + ".png"))
    #     print("\n")
    # output.eval_js('new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')


def ascend_txt(model, mgrid, c=28):
    model.train()
    out = model(mgrid)

    cutn = 256
    p_s = []
    for ch in range(cutn):
        size = torch.randint(int(0.5 * sideX), int(0.98 * sideX), ())
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx : offsetx + size, offsety : offsety + size]
        apper = torch.nn.functional.interpolate(
            apper, (224, 224), mode="bilinear"
        )
        p_s.append(nom(apper))
    into = torch.cat(p_s, 0)
    iii = imgnet(into)
    return -iii[:, c].mean()


def train(model, optimizer, mgrid, i, c=28):
    loss = ascend_txt(model, mgrid, c=c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i > 0 and i % 150 == 0:
        checkin(model, loss)


if __name__ == "__main__":
    imgnet = get_imagenet_model().to(device)

    nom = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    train_weights_all = []
    if os.path.exists("weights.pth"):
        train_weights_all = torch.load("weights.pth")
        print("Loaded weights.pth")
    for c in range(51, 5000):
        print(c)
        step = 0.0
        funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
            torch.cos(x + step + np.pi * 0.5)
        )
        model3 = Siren(2, 24, 8, 3, act_fn=funky_step).to(device)
        mgrid = get_mgrid(sideX)
        optimizer = torch.optim.Adam(model3.parameters(), 0.0001)
        for i in tqdm(range(150 * 10)):
            # step = np.clip(np.random.normal() * 0.7, -2, 2) * np.pi
            # funky_step = lambda x: torch.relu(torch.cos(x + step)) - torch.relu(
            #     torch.cos(x + step + np.pi * 0.5)
            # )
            train(model3, optimizer, mgrid, i, c=c)
        train_weights_all.append(model3.state_dict())
        torch.save(train_weights_all, "weights.pth")

    mgrid = get_mgrid(1920)

    for weights in train_weights_all:
        model3 = Siren(2, 24, 8, 3, act_fn=funky_step).cuda()
        model3.load_state_dict(weights)
        with torch.no_grad():
            r = forward(model3.cuda(), mgrid.cuda())
        im = r.cpu()[0][:, :1080, :1920]
        # displ(im)
