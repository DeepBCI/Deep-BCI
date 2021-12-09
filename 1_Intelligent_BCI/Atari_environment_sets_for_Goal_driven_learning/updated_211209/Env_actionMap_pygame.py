
import gym
import time
from psychopy import visual, core, event, monitors
from psychopy.hardware import keyboard
import time
import numpy as np
from scipy import io
import os
import math
import numpy as np
import itertools
import random
from gym import spaces
# import egi
# import egi.simple as egi
import egi3.simple as egi
# import egi.threaded as egi
import sys # sys.argv[]
import pickle
from datetime import datetime


def get_actionSpace(env_name='MountainCar-v0'):

    env = gym.make(env_name)
    num_action_space = env.action_space.n

    if env_name == 'CartPole-v1':
        action_space = {"right": 1, "left": 0}

    elif env_name == 'MountainCar-v0':
        action_space = {"right": 2, "left": 0}

    elif env_name == 'Acrobot-v1' :
        action_space = {"right": 0, "left": -1}

    # atar:
    # action_space = {'w': 2, 's': 5, 'a': 4, 'd': 3, ' ': 1}

    # if num_action_space == 2 :
    #     action_space = {'a': 0, 'd': 1} #left = 0, right =1, nothing = 2
    #
    # elif num_action_space == 3:
    #     action_space = {'a': 0, 'd': 1, 's': 2}

    # elif num_action_space > 3: # atari
    elif env_name == 'Pong-v0':
        action_space = {'w': 2, 's': 5, 'a': 4, 'd': 3, 'l': 1}

    else: # atari except pong
        meaning = env.get_action_meanings()  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        my_map = env.get_keys_to_action()  # {(): 0, (32,): 1, (100,): 2, (97,): 3, (32, 100): 4, (32, 97): 5}
        action_to_key = {v: k for k, v in my_map.items()}
        action_space = {}
        for i in range(num_action_space):
            if len(action_to_key[i]) == 1:
                if chr(action_to_key[i][0]) == ' ':
                    action_space.update({'l': i})
                else:
                    action_space.update({chr(action_to_key[i][0]): i})
            elif len(action_to_key[i]) == 2:
                if chr(action_to_key[i][0]) == ' ':
                    k1 = 'l'
                else:
                    k1 = chr(action_to_key[i][0])

                if chr(action_to_key[i][1]) == ' ':
                    k2 = 'l'
                else:
                    k2 = chr(action_to_key[i][1])
                k = [k1, k2]
                k.sort()
                action_space.update({tuple(k): i})
                # action_space.update({chr(action_to_key[i][1]): i})

            elif len(action_to_key[i]) == 3:
                if chr(action_to_key[i][0]) == ' ':
                    k1 = 'l'
                else:
                    k1 = chr(action_to_key[i][0])

                if chr(action_to_key[i][1]) == ' ':
                    k2 = 'l'
                else:
                    k2 = chr(action_to_key[i][1])
                if chr(action_to_key[i][2]) == ' ':
                    k3 = 'l'
                else:
                    k3 = chr(action_to_key[i][2])
                k = [k1, k2, k3]
                k.sort()
                action_space.update({tuple(k): i})
                # action_space.update({chr(action_to_key[i][1]): i})

        if env_name == 'Enduro-v0':
            action_space.pop('s')
            action_space.pop(('d', 's'))
            action_space.pop(('a', 's'))

        # print(action_space)


    return action_space


# setting environment
#
# env_name = 'Pong-v0'
# env = gym.make(env_name)
#
#
#
# action_space2 = get_actionSpace(env_name)

# print(action_space2)