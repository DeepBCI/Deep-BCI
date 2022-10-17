
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

    # cand_list = [[1, 2, 3, 1, 2, 3]] # TODO 바꾸기
    # cand_list = [[3, 1, 2, 3, 1, 2]]
    # cand_list = [[8,8,8,8,8,8]]
    cand_list = [[7,7,7,7,7,7]] # pitfall
    # cand_list = [[6,6,6,6,6,6]] # breakout
    # cand_list = [[2,1,3,2,1,3]]

    # uncertainty_list = random.sample(complexity_list, len(complexity_list)) # high = 1, low = 0

    # complexity_list = [[0, 0, 0, 1, 1, 1]] # TODO 바꾸기
    complexity_list = [[1, 1, 1, 1, 1, 1]]
    # complexity_list = [[0, 0, 0, 0, 0, 0]]

    # uncertainty_list = [[0, 0, 0, 1, 1, 1]]
    uncertainty_list = [[0, 0, 0, 0, 0, 0]]

    schedule_list = {'env_list': cand_list, 'complexity_list': complexity_list,
                       'uncertainty_list': uncertainty_list}  # session마다 COND_name_


    return schedule_list



def goal_setting_permut(is_main = 1):
    # for complexity schedule per one block

    random.seed(2023)
    # for HIGH complexity condition
    prev_list_complex = list(range(12)) + list(range(12))
    list_complex_comp = [x + 2 for x in prev_list_complex]
    list_complex_comp = random.sample(list_complex_comp, 13)
    list_complex = list_complex_comp + [1] * 7 + [0] * 7

    # for LOW complexity condition
    list_simple = [1] * 20 + [0] * 20

    # randomize schedule

    # for complex case
    random.seed(2024)
    complexity_list = []

    for t in range(800):
        list_sh = random.sample(list_complex, len(list_complex))
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
        complexity_list.pop(d)  # 499개

    # for simple case
    random.seed(2025)
    complexity_list_s = []

    for t in range(50000):
        list_sh = random.sample(list_simple, len(list_simple))
        if list_sh not in complexity_list_s:
            complexity_list_s.append(list_sh)

    del_index_s = []
    for i in range(len(complexity_list_s)):
        flag = 0
        for j in range(len(complexity_list_s[i])):
            try:
                if complexity_list_s[i][j] == complexity_list_s[i][j + 1] == complexity_list_s[i][j + 2]:
                    flag = 1
                    break
            except:
                continue
        if flag == 1:
            del_index_s.append(i)

    del_index_s.reverse()
    for d in del_index_s:
        complexity_list_s.pop(d)  # 39 개

    # change the name of stages

    list_complexity_LOW = complexity_list_s  # 39 개

    list_complexity_HIGH = []  # 499개


    for comp in complexity_list:
        list_complexity_HIGH_per_comp = []

        for num in comp:
            if num == 0:
                list_complexity_HIGH_per_comp.append(0)
            elif num == 1:
                list_complexity_HIGH_per_comp.append(1)
            elif num == 2:
                list_complexity_HIGH_per_comp.append(100)
            elif num == 3:
                list_complexity_HIGH_per_comp.append(101)
            elif num == 4:
                list_complexity_HIGH_per_comp.append(111)
            elif num == 5:
                list_complexity_HIGH_per_comp.append(110)
            elif num == 6:
                list_complexity_HIGH_per_comp.append(200)
            elif num == 7:
                list_complexity_HIGH_per_comp.append(201)
            elif num == 8:
                list_complexity_HIGH_per_comp.append(211)
            elif num == 9:
                list_complexity_HIGH_per_comp.append(210)
            elif num == 10:
                list_complexity_HIGH_per_comp.append(300)
            elif num == 11:
                list_complexity_HIGH_per_comp.append(301)
            elif num == 12:
                list_complexity_HIGH_per_comp.append(311)
            elif num == 13:
                list_complexity_HIGH_per_comp.append(310)
        list_complexity_HIGH.append(list_complexity_HIGH_per_comp)

    schedule_list = {'high_complexity_per_trial': list_complexity_HIGH, 'low_complexity_per_trial': list_complexity_LOW}  # session마다 COND_name_

    return schedule_list