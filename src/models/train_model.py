# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from models.dqn import DQN
from features.open_ai_gym import use_cuda
from models.replay_memory import ReplayMemory
import torch.optim as optim
import random
import math
from features.open_ai_gym import FloatTensor
from features.open_ai_gym import LongTensor
from models.replay_memory import Transition
from features.open_ai_gym import ByteTensor
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from features.open_ai_gym import Tensor
import matplotlib
from psutil import virtual_memory
matplotlib.use('Agg')
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display



BATCH_SIZE = 10
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
mem = virtual_memory()
print('memory toral {0}'.format(mem.total))
memory_size = mem.total // 80 // 3
print('memory size {0}'.format(memory_size))
memory = ReplayMemory(memory_size)

step_done = 0


def select_action(state):
    global step_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_done / EPS_DECAY)
    step_done += 1
    if sample > eps_threshold:
        # volatile
        #  set the inference mode
        #    http://pytorch.org/docs/master/notes/autograd.html
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


episide_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episide_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration [higher is better]')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        # unfold
        #    Returns a tensor which contains all slices of size size in the dimension dim.
        # view
        #    Returns a new tensor with the same data but different size.
        #        http://pytorch.org/docs/master/tensors.html
        # cat
        #    Concatenates the given sequence of seq Tensors in the given dimension.
        #        http://pytorch.org/docs/master/torch.html?highlight=cat#torch.cat
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.claer_output(wait=True)
        display.displae(plt.gcf())


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # gather
    #     http://pytorch.org/docs/master/torch.html?highlight=gather#torch.gather
    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    if next_state_values.volatile is None:
        next_state_values.volatile = False

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        # clamp_
        #   Clamp all elements in input into the range [min, max] and return a resulting Tensor.
        #       http://pytorch.org/docs/master/torch.html#torch.clamp
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
