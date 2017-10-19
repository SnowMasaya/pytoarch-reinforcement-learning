# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from models.train_model import memory
from models.train_model import select_action
from models.train_model import episide_durations
from models.train_model import plot_durations
from models.train_model import optimize_model
from features.open_ai_gym import Tensor
from visualization.input_extraction import env
from visualization.input_extraction import get_screen
from itertools import count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.ion()

last_sync = 0

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
plt.draw()
plt.savefig('Example_extracted_screen.png')

num_episodes = 100

for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        reward = Tensor([reward])

        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()
        if done:
            plt.pause(0.1)
            episide_durations.append(t + 1)
            plot_durations()
            break

print('complete')
# env.render(close=True)
# env.close()
plt.ioff()
plt.show()
plt.draw()
plt.savefig('reinforcement.png')

