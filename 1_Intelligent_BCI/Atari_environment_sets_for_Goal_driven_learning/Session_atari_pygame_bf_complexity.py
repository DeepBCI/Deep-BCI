import gym
import gym_task
import time
from psychopy import visual, core, event, monitors
from psychopy.hardware import keyboard as psychokeyboard

import keyboard as pykb
from multiprocessing import Process

import time
import numpy as np
from scipy import io
import os
import math
import numpy as np
import itertools
import random
# from my_scheduler_v3 import *
from Env_actionMap_pygame import *
from my_Scheduler_atari_pygame import *
from gym import spaces
# import egi
# import egi.simple as egi
import egi3.simple as egi
# import egi.threaded as egi
import sys  # sys.argv[]
import pickle
from datetime import datetime
from PIL import Image
import pygame

time_now = datetime.now().strftime('%Y%m%d-%H%M')


def print_pressed_keys(e):
    #line = ', '.join(str(code) for code in pykb._pressed_events)
    # '\r' and end='' overwrites the previous line.
    # ' '*40 prints 40 spaces at the end to ensure the previous line is cleared.
    #print('\r' + line + ' ' * 40, end='')
    pass

def keyboard_recording() :

    pykb.hook(print_pressed_keys)
    pykb.wait()

def openAI_function_atari(PART_NUMBER, SESS_NUMBER, ns=None, FrameRate=60,
                          day=1):  # part_number = 0-6  , env_name='Skiing-v0', slowing=0,

    p = Process(target=keyboard_recording)
    p.start()

    ## for session conditioning ##
    if SESS_NUMBER == 0:
        #### get schedule ####
        # if day == 1:
        #     num_eachGame = 2
        # else:
        #     num_eachGame = 2
        num_eachGame = 2

        cand_list = block_permut(num_of_gameType=3, num_eachGame=num_eachGame)

        direc = './SESS_atari'

        if not os.path.exists(direc):
            os.makedirs(direc)
        # pdb.set_trace()
        with open(direc + '/SESS_day' + str(day) + '.pkl', 'wb') as f:
            pickle.dump(cand_list, f)

        return cand_list

    else:
        #### load necessary data ####
        direc_SESS = './SESS_atari/SESS_day' + str(day) + '.pkl'

        with open(direc_SESS, 'rb') as f:
            SESSs = pickle.load(f)

        session = SESS_NUMBER - 1 # 0 1 2 3
        schedule_index_env = (3 * PART_NUMBER + session) % len(SESSs["env_list"])
        schedule_index_uncertainty = (3 * PART_NUMBER + session) % len(SESSs["uncertainty_list"])

        schedule_env = SESSs["env_list"][schedule_index_env]
        if PART_NUMBER % 2 == 0 :
            if session == 1: #day 1
                schedule_env.append(4)
            elif session == 2: #day 2
                schedule_env.append(5)

        else:
            if session == 1: #day 1
                schedule_env.append(5)
            elif session == 2: #day 2
                schedule_env.append(4)

        schedule_complexity = SESSs["complexity_list"][schedule_index_uncertainty]
        schedule_complexity.append(0)  # for few shot
        schedule_uncertainty = SESSs["uncertainty_list"][schedule_index_uncertainty]
        schedule_uncertainty.append(0)  # for few shot


        slow = {'Enduro-v0': 1, 'SpaceInvaders-v0': 1, 'MsPacman-v0': 2, 'Seaquest-v0': 1, 'Asterix-v0': 2, 'Kangaroo-v0': 1}

        block_num = 0
        run_time_min = 8 #8

        #### setting window ####
        white = (255, 255, 255)
        black = (0, 0, 0)
        X = 1920
        Y = 1080
        window_size = [1920, 1080]

        for block in schedule_env:
            block_complexity = schedule_complexity[block_num]
            block_uncertainty = schedule_uncertainty[block_num]
            block_num += 1

            if block_uncertainty == 1: #high
                # pp = [0.5, 0.5]
                # pp = [0.3, 0.7] # TODO: 바꾸기
                pp = [1, 0]
            else : #low
                # pp = [0.9, 0.1] # TODO: 바꾸기
                pp = [1, 0]

            t1 = core.Clock()
            t2 = core.Clock()
            t3 = core.Clock()  # for rt and ot
            t_ep = core.Clock()  # for first episode time
            t_block = core.Clock()  # for 8 min duration

            observations_sess = []
            info_keys_sess = []
            ots_sess = []
            reward_sess = []
            reward_per_image_sess = []

            t_ep.reset()
            t_block.reset()

            i_episode = 0

            """rest 1min"""

            t_done1 = time.time()

            t_leftover = core.Clock()  # for calculate rest time
            t_leftover.reset()

            while t_block.getTime() <= 60 * run_time_min:  # 8 min

                if block == 1:
                    env_name = 'Seaquest-v0'
                    # env_name = 'MsPacman-v0'
                elif block == 2:
                    env_name = 'MsPacman-v0'
                elif block == 3:
                    env_name = 'SpaceInvaders-v0'
                elif block == 4:
                    env_name = 'Asterix-v0'
                elif block == 5:
                    env_name = 'Kangaroo-v0'

                # setting environment
                # env = gym.make('f' + env_name)
                env = gym.make(env_name)
                action_space = get_actionSpace(env_name)
                key_list = list(action_space.keys()) + ['h']
                slowing = slow[env_name]
                observation = env.reset()

                info_keys = []
                ots = []

                if i_episode == 0 and block_num == 1:

                    if ns is not None:
                        ns.StartRecording()

                    pygame.init()
                    display_surface = pygame.display.set_mode((X, Y), display=0)  # 검정화면 뜸
                    pygame.display.set_caption('Show Text')
                    font1 = pygame.font.Font('freesansbold.ttf', 64)
                    text = font1.render('Press the "t" button to start the game.', True, white, black)
                    textRect = text.get_rect()
                    textRect.center = (X // 2, Y // 2)

                    ots.append(t_ep.getTime())  # meassage time point

                    display_surface.fill(black)
                    display_surface.blit(text, textRect)
                    pygame.display.update()


                    while True:

                        if pykb.is_pressed(20):
                            break


                        for event in pygame.event.get():

                            if event.type == pygame.QUIT:
                                running = False

                    del font1
                    pygame.quit()


                elif i_episode == 0 and block_num != 1:
                    # rest 끝나기 30초 전에 start recording
                    ots.append(
                        t_ep.getTime())  # fixation time point  # TODO: 사수님 이거 여기 넣어야 할까요, start recording 뒤에 넣어야할까요...

                    t_rec3 = time.time()
                    if t_rec1 in locals():
                        print(t_rec3 - t_rec1)
                        if t_rec3 - t_rec1 < 30:
                            time.sleep(30 - (t_rec3 - t_rec1))

                    if ns is not None:
                        ns.StartRecording()

                    # ots.append(t_ep.getTime()) # fixation time point

                    # 2분 초과해도 30초는 쉬게 해주기
                    time.sleep(30)


                else:
                    if ns is not None:
                        ns.StartRecording()

                    ots.append(t_ep.getTime())  # fixation time point
                    time.sleep(0.5 + np.random.rand())

                observation = env.reset()

                env.render()


                ots.append(t_ep.getTime())  # # episode 시작 time point

                if ns is not None:
                    ns.send_event('epi' + str(i_episode), label="episode", timestamp=egi.ms_localtime())

                ## 시작점
                observations = []
                reward_per_images = []
                t1.reset()
                t2.reset()
                t3.reset()
                t_refresh1 = time.time()
                n = 0
                t = 0
                tt = 0
                t0 = time.time()
                done = False
                reward_c = 0
                pygame.quit()

                while True:  # sampling

                    tt += 1
                    info_key = []

                    if n == 0:
                        t2.reset()

                    gt2 = t2.getTime()

                    if gt2 > 1:
                        t_refresh2 = time.time()
                        if int(t_refresh2 - t_refresh1) % 10 == 0:
                            print(f"refresh rate is {n} Hz.")
                            print("Rest time for block_{0} is {1} min.".format(block_num,
                                                                               int(
                                                                                   run_time_min - t_leftover.getTime() / 60)))  # 8-~~
                        n = 0
                        t2.reset()

                    action = 0
                    gt = t1.getTime()

                    if gt > (slowing / FrameRate):  # render 60Hz   #0.01694915
                        keyss = []
                        if env_name != 'Enduro-v0' and env_name != 'SpaceInvaders-v0':
                            if pykb.is_pressed(31):
                                keyss.append('s')
                            if pykb.is_pressed(17):
                                keyss.append('w')
                        if pykb.is_pressed(32):
                            keyss.append('d')
                        if pykb.is_pressed(30):
                            keyss.append('a')

                        if pykb.is_pressed(38):
                            keyss.append('l')
                        if pykb.is_pressed(35): # dlrj
                            keyss.append('h')

                        reaction_time = t3.getTime()

                        if len(keyss) > 1:
                            if 'a' in keyss and 'd' in keyss:
                                keyss.remove('a')
                                keyss.remove('d')

                            if 'w' in keyss and 's' in keyss:
                                keyss.remove('w')
                                keyss.remove('s')

                        final_len = len(keyss)


                        # for random key for uncertainty
                        ll = list(action_space.keys())

                        if 'l' not in keyss:
                            # new random key
                            del_list = []

                            for key in ll:
                                l_key = list(key)
                                if 'l' in l_key:
                                    del_list.append(key)

                            for i in del_list:
                                ll.remove(i)

                        ll_random = random.sample(ll, 1)
                        action_random = action_space[ll_random[0]]




                        if final_len == 0:  # len(keys) == 0:  # press nothing


                            # if ns is not None:
                            #     ns.send_event('act' + str(t), label='None',
                            #                   timestamp=egi.ms_localtime())  # label : left, right,
                            uncertainty = np.random.choice(2, 1, p=pp)

                            if uncertainty == 0:
                                info_key.append(np.nan)  # action action
                                info_key.append(np.nan)  # rt
                                # info_key.append(np.nan)  # duration
                                # info_key.append(np.nan)  # common/rare
                                info_key.append(uncertainty) # block_uncertainty...?
                                info_keys.append(info_key)
                                action = 0

                            if uncertainty == 1:
                                info_key.append(np.nan)  # action action
                                info_key.append(np.nan)  # rt
                                # info_key.append(np.nan)  # duration
                                # info_key.append(np.nan)  # common/rare
                                info_key.append(uncertainty)
                                info_keys.append(info_key)

                                action = action_random


                        elif final_len == 1:
                            if ns is not None:
                                ns.send_event('act' + str(t), label=keyss[0],
                                              timestamp=egi.ms_localtime())  # label : left, right,

                            if keyss[0] == 'h':
                                return



                            if keyss[0] in action_space.keys():
                                uncertainty = np.random.choice(2, 1, p=pp) # 0, 1중 1개 pp = [0.9, 0.1] or [0.5, 0.5]의 확률로
                                # uncertainty = 0

                                if uncertainty == 0:  # condition normal

                                    action = action_space[keyss[0]]

                                    info_key.append(keyss[0])
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append(dd) #duration
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                                if uncertainty == 1:  # condition not normal

                                    true_action = action_space[keyss[0]]
                                    # action = 0 # drop button
                                    action = action_random

                                    info_key.append(keyss[0])
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append(dd) #duration
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                        elif final_len == 2:  # : # press two key
                            if ns is not None:
                                ns.send_event('act' + str(t), label="({0}, {1})".format(keyss[0], keyss[1]),
                                              timestamp=egi.ms_localtime())  # label : left, right,

                            action_pre = [keyss[0], keyss[1]]
                            action_pre.sort()

                            if tuple(action_pre) in action_space.keys():

                                # uncertainty = 0
                                uncertainty = np.random.choice(2, 1, p=pp)  # 0, 1중 1개 pp = [0.9, 0.1] or [0.5, 0.5]의 확률로

                                if uncertainty == 0:  # condition common

                                    action = action_space[tuple(action_pre)]
                                    # print(action_pre)
                                    # k1_dur = action_inorder[0].duration
                                    # k2_dur = action_inorder[1].duration
                                    info_key.append((keyss[0], keyss[1]))
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append((dd1, dd2))
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                                if uncertainty == 1:  # condition not normal

                                    true_action = action_space[tuple(action_pre)]
                                    # action = 0 # drop button
                                    action = action_random

                                    info_key.append((keyss[0], keyss[1]))
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append((dd1, dd2))
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                            else:  # combination 안될 때 last key만
                                print("no comb")
                                uncertainty = np.random.choice(2, 1, p=pp)

                                if uncertainty == 0:  # condition common
                                    action = action_space[keyss[0]]

                                    info_key.append(keyss[0])
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append(dd)
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                                if uncertainty == 1:  # condition not normal

                                    true_action = action_space[keyss[0]]
                                    # action = 0 # drop button
                                    action = action_random

                                    info_key.append(keyss[0])
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append(dd)
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                        else:  # len(keys) >= 3

                            if ns is not None:
                                ns.send_event('act' + str(t),
                                              label="({0}, {1}, {2})".format(keyss[0], keyss[1], keyss[2]),
                                              timestamp=egi.ms_localtime())  # label : left, right,

                            action_pre = [keyss[0], keyss[1], keyss[2]]
                            action_pre.sort()
                            if tuple(action_pre) in action_space.keys():
                                # uncertainty = np.random.choice(2, 1, p=p)
                                uncertainty = np.random.choice(2, 1, p=pp)
                                # uncertainty = 0
                                # print('Len buttons: '+str(len(keys)))
                                # print('Long press: '+keys[-1].name+keys[-2].name)

                                if uncertainty == 0:  # condition common

                                    action = action_space[tuple(action_pre)]

                                    info_key.append((keyss[0], keyss[1], keyss[2]))
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append((dd1, dd2, dd3))
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)
                                    #
                                    # k1_dur = action_inorder[0].duration
                                    # k2_dur = action_inorder[1].duration
                                    # k3_dur = action_inorder[2].duration
                                if uncertainty == 1:  # condition not normal
                                    true_action = action_space[tuple(action_pre)]
                                    # action = 0  # drop button
                                    action = action_random

                                    info_key.append((keyss[0], keyss[1], keyss[2]))
                                    info_key.append(reaction_time)  # keys[-1].rt)
                                    # info_key.append((dd1, dd2, dd3))
                                    info_key.append(uncertainty)
                                    info_keys.append(info_key)

                            else:  # combination 안될 때 last key만
                                print("no comb in 3")
                                uncertainty = np.random.choice(2, 1, p=pp)
                                # uncertainty = 0
                                if 'l' in action_pre:
                                    # l_dur = 0

                                    action_pre.remove('l')
                                    for i in range(len(keyss)):
                                        if keyss[i] == 'l':
                                            keyss.pop(i)
                                            break

                                    if uncertainty == 0 :
                                        actual_act = [keyss[0], 'l']
                                        actual_act.sort()
                                        action = action_space[tuple(actual_act)]

                                        info_key.append((keyss[0], 'l'))
                                        info_key.append(reaction_time)  # keys[-1].rt)
                                        # info_key.append((dd, dl))
                                        info_key.append(uncertainty)
                                        info_keys.append(info_key)

                                    if uncertainty == 1:
                                        actual_act = [keyss[0], 'l']
                                        actual_act.sort()
                                        true_action = action_space[tuple(actual_act)]
                                        # action = 0 # drop button
                                        action = action_random

                                        info_key.append((keyss[0], 'l'))
                                        info_key.append(reaction_time)  # keys[-1].rt)
                                        # info_key.append((dd, dl))
                                        info_key.append(uncertainty)
                                        info_keys.append(info_key)


                                else:  # action_pre 에 'l'없을때
                                    if uncertainty == 0:
                                        action = action_space[keyss[0]]
                                        # print(action_inorder[0].name)

                                        info_key.append(keyss[0])
                                        info_key.append(reaction_time)  # keys[-1].rt)
                                        # info_key.append(dd)
                                        info_key.append(uncertainty)
                                        info_keys.append(info_key)

                                    if uncertainty == 1:
                                        true_action = action_space[keyss[0]]
                                        # action = 0  # drop button
                                        action = action_random
                                        # print(action_inorder[0].name)

                                        info_key.append(keyss[0])
                                        info_key.append(reaction_time)  # keys[-1].rt)
                                        # info_key.append(dd)
                                        info_key.append(uncertainty)
                                        info_keys.append(info_key)

                        observation, reward, done, info = env.step(action)
                        ots.append(t3.getTime())  # observation time point

                        reward_c += reward
                        observations.append(observation)
                        reward_per_images.append(reward)

                        env.render()
                        n += 1
                        t1.reset()
                        t += 1

                    t_done2 = time.time()

                    if done or t_done2 - t_done1 > run_time_min * 60:  # 8*60
                        # print("Episode finished after {} frames".format(t))  # time step t
                        # print("Episode finished after {} time steps".format(tt))  # time step t
                        # print(f"rendr times {gt5} time steps") #time step t
                        break

                # after all time step
                # every one episode
                env.close()
                # pygame.display.quit()
                pygame.quit()



                pygame.init()

                display_surface = pygame.display.set_mode((X, Y), display=0)  # 검정화면 뜸
                pygame.display.set_caption('Show Text')
                font2 = pygame.font.Font('freesansbold.ttf', 64)
                msg = '{} points'.format(int(reward_c))
                text = font2.render(msg, True, white, black)
                textRect2 = text.get_rect()
                textRect2.center = (X // 2, Y // 2)
                # reward 보여줄 때 event tagging
                if ns is not None:
                    ns.send_event('reward_epi' + str(i_episode), label="reward", timestamp=egi.ms_localtime())

                t4 = time.time()
                display_surface.fill(black)
                display_surface.blit(text, textRect2)
                pygame.display.update()

                ots_sess.append(ots)
                reward_sess.append(int(reward_c))
                info_keys_sess.append(info_keys)
                observations_sess.append(observations)
                reward_per_image_sess.append(reward_per_images)

                i_episode += 1

                frame_data_dict = {'observation_time': ots_sess, 'reward_c': reward_sess,
                                   'rewards_per_image': reward_per_image_sess,
                                   'key': info_keys_sess, 'session': schedule_env,
                                   'session_tag': "1: Seaquest, 2: MsPacman, 3: SpaceInvaders, 4: Asterix, 5: Kangaroo",
                                   'complexity_schedule': schedule_complexity, 'uncertainty_schedule': schedule_uncertainty}  # session마다 COND_name_

                # for saving
                env_name2 = env_name.split('-')[0]
                direc_save = './result_save/ATARI' + '/Subject{0}/session{1}_'.format(
                    PART_NUMBER, SESS_NUMBER) + time_now  # + "/episode{0:03d}_".format(i_episode - 1) + env_name2

                if not os.path.exists(direc_save):
                    os.makedirs(direc_save)

                io.savemat(direc_save + f'/block{block_num}_' + env_name2 + time_now + '.mat', frame_data_dict) #TODO: 돌려놔
                t5 = time.time()

                # DK
                svtime = t5 - t4
                print(str(svtime/60)+"min to save data / one trial")

                if 2 > svtime: #2*60
                    time.sleep(2 - svtime + np.random.rand())

                del font2 #TODO
                pygame.quit()

            pygame.init()
            # pygame.display.init()

            display_surface = pygame.display.set_mode((X, Y), display=0)  # 검정화면 뜸
            pygame.display.set_caption('Show Text')
            font3 = pygame.font.Font('freesansbold.ttf', 64)

            msg3_1 = "Total Score is {} points.".format(sum(reward_sess))
            msg3_2 = "Please ready for the next game!"
            text1 = font3.render(msg3_1, True, white, black)
            text2 = font3.render(msg3_2, True, white, black)
            textRect31 = text1.get_rect()
            textRect32 = text2.get_rect()
            textRect31.center = (X // 2, Y // 2 - Y // 20)
            textRect32.center = (X // 2, Y // 2 + Y // 20)
            display_surface.fill(black)
            display_surface.blit(text1, textRect31)
            display_surface.blit(text2, textRect32)
            pygame.display.update()


            t_rest = core.Clock()  # for rest
            t_rest.reset()

            # reward 보여줄 때 event tagging
            if ns is not None:
                ns.send_event('reward_block' + str(block_num), label="total_r", timestamp=egi.ms_localtime())

            # reward 보여주고 5초 뒤에 stop recording
            time.sleep(5)

            if ns is not None:
                ns.StopRecording()

            t_saving1 = time.time()  # for rest

            ##### save obs img #####
            env_name2 = env_name.split('-')[0]

            for i_episodei in range(len(observations_sess)):
                direc_img = "./result_save/ATARI" + '/Subject{0}/session{1}_'.format(
                    PART_NUMBER, SESS_NUMBER) + time_now + "/block{}_".format(
                    block_num) + env_name2 + "/episode{0:03d}".format(i_episodei)

                if not os.path.exists(direc_img):
                    os.makedirs(direc_img)

                observations_i = observations_sess[i_episodei]
                for ob in range(len(observations_i)):
                    Image.fromarray(observations_i[ob]).save(
                        direc_img + "/obs_{0:03d}.png".format(ob))

            t_saving2 = time.time()
            print("time for saving images per block: {} ".format(t_saving2 - t_saving1))
            # ns.sync()
            # ns.EndSession()
            # ns.finalize()
            """rest 1min"""
            t_restornot = t_rest.getTime()
            rest_time = 60  # + np.random.rand()  # 60 sec
            if t_restornot < rest_time:
                time.sleep(rest_time - t_restornot)

            t_rec1 = time.time()

            del font3
            pygame.quit()

            pygame.init()
            print("press t to continue.")
            while True:
                if pykb.is_pressed(20):
                    pygame.quit()
                    break

                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        running = False
            print("next")
    # else:
    #     raise ValueError
    p.join()
    print("finish")
    return 0



