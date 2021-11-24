
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

def block_permut(num_of_gameType = 3, num_eachGame = 2):
    """한 7개 만들어놔서 돌려쓸 생각"""
    DAY_SESSION = []

    len_sess = num_eachGame * num_of_gameType

    ### initialize : 112233 ###
    for game_type in range(num_of_gameType) :
        for i in range(num_eachGame) :
            DAY_SESSION.append(game_type+1)

    ### permutation :
    # in random order with the one constraint of never playing the same game twice in a row  ###

    # env_name randomize
    cand_list = []
    random.seed(2021)
    for t in range(100):  # make session candidates # 100
        res = random.sample(DAY_SESSION, len(DAY_SESSION))
        if res not in cand_list:
            flag = 0
            for i in range(len(res)):
                try:
                    if res[i] == res[i + 1]:
                        flag = 1
                except:
                    # print("last")
                    ''
                if flag == 1:
                    break
            if flag == 0 :
                cand_list.append(res)

    # complexity level randomize
    list = [1, 1, 1, 0, 0, 0]
    complexity_list = []  # high = 1, low = 0

    random.seed(2022)
    for t in range(100):
        list_sh = random.sample(list, len(list))
        # print(list_sh)
        if list_sh not in complexity_list:
            complexity_list.append(list_sh)

    del_index = []
    for i in range(len(complexity_list)):
        flag = 0
        for j in range(len(complexity_list[i])):
            try:
                if complexity_list[i][j] == complexity_list[i][j + 1] == complexity_list[i][j + 2]:
                    flag = 1
                    break
            except:
                continue
        if flag == 1:
            del_index.append(i)

    del_index.reverse()
    for d in del_index:
        complexity_list.pop(d)

    # uncertainty level randomize
    random.seed(2022)
    uncertainty_list = random.sample(complexity_list, len(complexity_list)) # high = 1, low = 0

    schedule_list = {'env_list': cand_list, 'complexity_list': complexity_list,
                       'uncertainty_list': uncertainty_list}  # session마다 COND_name_


    return schedule_list

