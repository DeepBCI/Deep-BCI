
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

    # dname randomize
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