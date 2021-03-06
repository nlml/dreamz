# from cppn_v1 import get_net
from cppn_v3 import get_net
from midi_stuff import get_rtmidi_obj, process_midi_msg_stack
from copy import deepcopy
import time
import example
import numpy as np
import torch
import torch.nn as nn
from dreamz.cppn import get_xy_mesh


MIDI_PORT = 2
MOMENTUM = 0.99
EXCESS = 0.5
TRANSITION_SPEED = 0.3
WARP_SPEED = 0.005
OTHER_DIM = 6


class PixelRenderer:
    def __init__(self, w, h):
        # Make the window
        self.w = w
        self.h = h
        example.setup(self.h, self.w, 0, 0)
        self.a = example.make_array(w * h * 3)
        self.last_frame_time = time.time()
        self.fps = 10.0
        self.frame_count = 0

    def render(self, im):
        self.frame_count += 1
        self.a = im.reshape(-1)
        example.render(self.a, self.w)
        self.fps = 0.9 * self.fps + 0.1 * 1.0 / (
            time.time() - self.last_frame_time
        )
        if self.frame_count % 10 == 0:
            print(round(self.fps, 2))
        self.last_frame_time = time.time()

    def shutdown(self):
        example.kill()


def get_targ():
    targ = np.random.rand(OTHER_DIM).astype(np.float32) * 2.0 - 1.0
    # print('new targ: ', targ)
    targ = torch.FloatTensor(targ).to(device)
    return targ


def interpolate_state_dicts(state_dicts, alpha):
    new_state_dict = deepcopy(state_dicts[0])
    for k in new_state_dict:
        new_state_dict[k] = (
            new_state_dict[k] * (1 - alpha) + state_dicts[1][k] * alpha
        )
    return new_state_dict


rtmidi_obj = get_rtmidi_obj(MIDI_PORT)

size = [108 * 1, 192 * 1]
# size = [108 * 2.513089005, 192 * 2.513089005]
size = [int(round(i)) for i in size]
mesh = get_xy_mesh(size)
device = "cuda" if torch.cuda.is_available() else "cpu"

state_dicts, viz, m = get_net(device)

mesh = mesh.to(device)

other = torch.FloatTensor([0.0] * OTHER_DIM).to(device)
grad = torch.FloatTensor([0.0] * OTHER_DIM).to(device)
grad_use = grad.clone()
targ = get_targ()


def get_actual_size():
    with torch.no_grad():
        out = m(mesh, other)
    return out.shape[1:3]


h, w = get_actual_size()
# h, w = 1080, 1920
print(h, w, "actual size")

current_sd_sel = list(range(1, len(state_dicts)))
sd_changes_counter = 0
next_sd_sel = sd_changes_counter % len(state_dicts)

curr_state_dict_interp_alpha = 0.5
curr_state_dict_interp_target = 1.0 + EXCESS
changed = False

renderer = PixelRenderer(w, h)

for i in range(50000):

    midi_events = process_midi_msg_stack(rtmidi_obj)
    if len(midi_events):
        # print(midi_events)
        if 36 in midi_events:
            curr_state_dict_interp_target *= -1.0
        for ev in midi_events:
            if isinstance(ev, tuple):
                if ev[0] == 3:
                    WARP_SPEED = ev[1] / 127.0 + 0.005

    curr_state_dict_interp_grad = (
        curr_state_dict_interp_target - curr_state_dict_interp_alpha
    )
    curr_state_dict_interp_alpha += (
        curr_state_dict_interp_grad * TRANSITION_SPEED
    )
    if curr_state_dict_interp_alpha >= 1.0:
        # curr_state_dict_interp_target = -1.0
        if not changed:
            viz.load_state_dict(
                interpolate_state_dicts(
                    [state_dicts[i_sd] for i_sd in current_sd_sel], 1.0
                )
            )
            current_sd_sel = [next_sd_sel, current_sd_sel[1]]
            sd_changes_counter += 1
            next_sd_sel = sd_changes_counter % len(state_dicts)
            changed = True
    elif curr_state_dict_interp_alpha <= -1.0:
        # curr_state_dict_interp_target = 1.0
        if not changed:
            viz.load_state_dict(
                interpolate_state_dicts(
                    [state_dicts[i_sd] for i_sd in current_sd_sel], 0.0
                )
            )
            current_sd_sel = [current_sd_sel[0], next_sd_sel]
            sd_changes_counter += 1
            next_sd_sel = sd_changes_counter % len(state_dicts)
            changed = True

    # if i % 1 == 0:
    #     print(curr_state_dict_interp_alpha, 'alpha')
    if (
        curr_state_dict_interp_alpha <= 1.0
        and curr_state_dict_interp_alpha >= -1.0
    ):
        changed = False
        viz.load_state_dict(
            interpolate_state_dicts(
                [state_dicts[i_sd] for i_sd in current_sd_sel],
                curr_state_dict_interp_alpha / 2 + 0.5,
            )
        )

    now = time.time()
    # a *= 0
    grad_use = (1 - MOMENTUM) * grad + MOMENTUM * grad_use
    other -= WARP_SPEED * grad_use
    with torch.no_grad():
        out = m(mesh, other)
    if torch.min(torch.abs(other - targ)) < 0.05:
        targ = get_targ()
    grad = other - targ
    to_render = (out[0].cpu().numpy() * 255).astype(np.uint8)
    renderer.render(to_render)
