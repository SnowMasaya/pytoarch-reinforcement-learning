# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import torchvision.transforms as T
from PIL import Image
import gym
import numpy as np
import torch
from features.open_ai_gym import Tensor


env = gym.make('CartPole-v0').unwrapped


resize = T.Compose([T.ToPILImage(),
                   T.Scale(40, interpolation=Image.CUBIC),
                   T.ToTensor()])

screen_width = 600


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0).type(Tensor)

