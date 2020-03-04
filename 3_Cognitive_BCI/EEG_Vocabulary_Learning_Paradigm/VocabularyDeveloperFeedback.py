# -*- coding: utf-8 -*-


'''
EEG-based Korean-German Vocabulary Study Paradigm
This is a public shared version of the paradigm code.
Several components have been omitted due to copyright concerns, including the main dictionary and some I/O dll files.
Note that this paradigm also requires Pyff Framework in order to run.
'''

import pygame
import numpy
import time
import sys, os
import random
import sqlite3
import win32api
import korin
from pygame.locals import *
import ctypes
import socket
from conversion import ncr_to_python
from conversion import ucn_to_python
import PyQt4

from FeedbackBase.MainloopFeedback import MainloopFeedback
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
win32api.LoadKeyboardLayout('E0010412', 1)


class VocabularyDeveloperFeedback(MainloopFeedback):
    """Feedback for showing pairs of vocabulary"""

    # Markers written to parallel port

    # Used Markers
    INIT_FEEDBACK = 200  # not recorded on db, but is sent to device on startup
    GAME_STATUS_PLAY = 20
    GAME_STATUS_PAUSE = 21
    GAME_OVER = 254

    TESTING = 150
    TESTING_CORRECT = 151  ##
    TESTING_INCORRECT = 152  ##
    TRAINING = 160
    TRAINING_CORRECT = 161  ##
    TRAINING_INCORRECT = 162  ##

    DISTRACTOR = 180  # newly added.. so as to not analyze distractor trials
    MATH_TEST = 190
    NEW_PAIR = 30
    TEST = 32
    # UPDATE = 112

    # unused markers (yet I'm not sure if it should be)
    FIRST_PRESS = 115  # reinstate
    LAST_PRESS = 116  # x

    # unused markers
    KNOWN_PAIR = 31
    WELCOME = 50
    FIXATION_PERIOD = 101  ###
    PRESENTATION_PERIOD = 102  ###
    INTER_PERIOD = 103  ###
    FIXATION_PERIOD_TEST = 201  ###
    PRESENTATION_PERIOD_TEST = 202  ###
    ANSWER_CORRECT = 171
    ANSWER_INCORRECT = 172
    INTER_PERIOD_TEST = 203
    POSSIBLE_LEARNING = 111  ###

    # JAN06 : distinguishable distractors, train-test ask difference

    DISTRACTOR_ASK = 181
    ASK_TRAIN = 34

    # PRESENT_WORD_01 = 90
    # PRESENT_WORD_02 = 91
    # PRESENT_WORD_03 = 92
    # PRESENT_WORD_04 = 93
    # PRESENT_WORD_05 = 94
    # PRESENT_WORD_05 = 94
    # PRESENT_WORD_06 = 95
    # PRESENT_WORD_07 = 96
    # PRESENT_WORD_08 = 97
    # PRESENT_WORD_09 = 98
    # PRESENT_WORD_10 = 99

    ###end of markers

    def init(self):
        """Called at the beginning of the Feedback's lifecycle.

        More specifically: in Feedback.on_init().
        """
        self.logger.debug("on_init")

        # In dev mode and don't want to go through adding user names? keep this as 1.
        self.pport_address = 0xC010

        self.devmode = 1
        self.username = 'VP'
        self.session_per_cycle = 4

        # used for logging ask_or_test (v2) and last event_key
        self.past_test_start = 0

        # time before training and after training for analysis purposes
        self.prePostTrain = 1  # 15
        # time before showing
        self.fixation_time = 1.25
        # time after key pressing
        self.inter_time = 0.25
        self.time_b4_enter_press = self.inter_time
        self.fixation_time_test = 1.25
        self.inter_time_test = 0.25
        self.user_feedback_time = 1
        self.final_result_time = 10
        self.instruction_show_time = 10
        self.fullscreen = False
        # self.fullscreen =  True
        # self.screenWidth =  2560
        # self.screenHeight =  1440
        self.screenWidth = 1400
        self.screenHeight = 700
        ## for self use only
        self.hostname = socket.gethostname()
        if self.hostname == 'sokrates':
            self.screenWidth = 1000
            self.screenHeight = 700
        self.backgroundColor = (255, 255, 255)
        self.cursorColor = (0, 0, 0)
        self.fontColor = self.cursorColor

        self.part = 1
        # print 'self.part'+str(self.part)
        self.learnt_words = 0
        self.show_display_limit = 10
        self.VP = 'VPaa'
        self.VPlog = 'Temp'
        # create a directory of VP if it does'nt exist
        # is(dir),if not then mkdir() etc etc
        self.lection_path = os.path.dirname(sys.modules[__name__].__file__) + '/'

        # check for the computer and set path accordingly

        self.st_path = ''
        self.st_path = 'SET_DIRECTORY_NAME_HERE'

    # self.store_path = ''
    self.store_path = '\\'.join([self.st_path, self.VP, ''])
    # self.store_path = self.st_path
    # self.store_path = '/home/sophie/Dokumente/HiWiJob/Data/'

    self.f = 0

    self.finnish = False
    self.multiple_choice = False

    # here is stored how often they have been remembered correctly
    self.nr_of_max_reps = 10
    self.nr_of_correct_reps = 3
    # self.max_dict_length = 60 # probably need to be removed for the new paradigm!!!
    self.nr_of_words_2_b_shown = 4
    self.distractor_size = 4  # is this enough?? or would 5-6 gap be good

    self.tot_words = 2000  # total number of words

    self.learn_words = 60  # words to learn in a session? update_bag uses this.

    self.bag = []
    self.distractor_bag = []
    self.distrctDict = []
    self.dict_indices = []  # to be read from the dictionary, 3rd column
    self.learnable_word_count = 0
    self.known_pair_index = []

    self.distractor_file = 'lektion_distractor.txt'

    self.minBagSize = self.distractor_size  # else 5??

    # database for loggging

    self.presentation_time = 5  # 5 PRESENTATION TIME
    self.sessionid = 0;
    self.conn = sqlite3.connect('\\'.join([self.st_path, 'memory_ex.db']))
    self.c = self.conn.cursor()
    self.create_db()
    # toggle var for korean input
    self.krtoggle = 1

    self.instruction_file = "instruction.txt"
    self.instruction_path = ''.join([self.lection_path, self.instruction_file])

    self.initial_bag_size = 10
    self.bag_size = self.initial_bag_size
    self.bag_filling_index = 0
    self.showed_sequence = []
    self.asked_sequence = []

    self.maths_filename = 'maths.txt'
    self.maths_questions = self.make_maths_questions(''.join \
                                                         ([self.lection_path, self.maths_filename]))
    self.Filenames = 'lession_database.txt'
    self.distrctDict = []
    if self.finnish:
        self.Filenames = ['Lektion_Finnisch_Kommunikation.txt']
    print
    "+++++++++++++++++++++++++++++++++++++++++++++++"
    print
    "VocabularyFeedbackDeveloper HAS BEEN INITIALIZED FROM PYFF"
    print
    "***********************************************"
    self.send_parallel_and_write(self.INIT_FEEDBACK)


def pre_mainloop(self):  # check with Sophie (parts have changed so be careful)

    # if not(self.part==3) and not(self.part==6) and not(self.part==7):
    #
    #     self.distractor_part = self.part
    #     if self.part>3 and self.part<6:
    #         self.distractor_part = self.part - 1
    #         # distractor has only 4 parts so -1 takes 4 and 5 to 3 and 4 respectively
    self.distr_show = []
    #     self.distractor_bag = range( (self.distractor_part-1) * 5 + 241 , (self.distractor_part-1)* 5 + 246)
    #     temp_dict = self.distrctDict[(self.distractor_part-1) * 5 : (self.distractor_part-1)* 5 + 5]
    self.distrctDict = []
    #     self.distrctDict = temp_dict

    """Called before entering the mainloop, e.g. after on_play."""
    self.init_pygame()
    self.init_graphics()

    self.welcome()
    self.ask_name()

    print
    'db store_path: ' + str(self.st_path)
    # create logfile with run time for future reference
    localtime = time.localtime()
    # print 'self.path'+str(self.part)
    # if self.part==1 or self.part==2 or self.part==4 or self.part==5:
    #     typSes = 'train'
    # elif self.part==3 or self.part==6:
    #     typSes = 'test'
    self.c.execute('SELECT COUNT(*) FROM session WHERE username=?', (self.username,))
    self.session_num = self.c.fetchone()[0]

    # if self.session_num % 3 == 1:
    #     self.typSes = 'Train_A'
    # elif self.session_num % 3 == 2:
    #     self.typSes = 'Train_B'
    # elif self.session_num % 3 == 0:
    #     self.typSes = 'Test'

    # self.typSes = 'Train_A'

    self.lession_num = divmod(self.session_num, self.session_per_cycle)[1]
    if self.lession_num == 1:
        self.typSes = 'Train_A'
    elif self.lession_num == 2:
        self.typSes = 'Train_B'
    elif self.lession_num == 3:
        self.typSes = 'Train_C'
    else:
        self.typSes = 'Test'

    self.db_query('update session set session_type = ? where sessionid = ?', (self.typSes, self.sessionid))

    print
    'CYCLE # ', self.lession_num
    # self.db_query('update session set username = ? where sessionid = ?', (self.username, self.sessionid))

    print
    'Session No.', self.session_num, ' ', self.typSes
    #

    [self.distrctDict, self.distractor_idx] = self.make_dictionary(''.join \
                                                                       ([self.lection_path, self.distractor_file]))

    # chnge the make_dictionary to include indices as well
    # to obtain the necessary indices for the distractor bag for each part

    "allocate distractor part"
    if self.typSes == 'Train_A':
        # self.distractor_part =  (self.lession_num - 1) * 2 + 1
        self.distractor_bag = self.distractor_idx[((self.lession_num - 1) * 10 + 1): ((self.lession_num - 1) * 10 + 6)]
        self.distrctDict = self.distrctDict[((self.lession_num - 1) * 10 + 1): ((self.lession_num - 1) * 10 + 6)]
    elif self.typSes == 'Train_B':
        # self.distractor_part =  (self.lession_num - 1) * 2 + 2
        self.distractor_bag = self.distractor_idx[((self.lession_num - 1) * 10 + 6): (self.lession_num * 10 + 1)]
        self.distrctDict = self.distrctDict[((self.lession_num - 1) * 10 + 6): (self.lession_num * 10 + 1)]
    elif self.typSes == 'Train_C':
        # self.distractor_part =  (self.lession_num - 1) * 2 + 3
        self.distractor_bag = self.distractor_idx[((self.lession_num) * 10 + 1): ((self.lession_num) * 10 + 6)]
        self.distrctDict = self.distrctDict[((self.lession_num) * 10 + 1): ((self.lession_num) * 10 + 6)]
    # print 'distractor_bag: ', self.distractor_bag
    # print 'distractor_dict: ', self.distrctDict


def post_mainloop(self):
    """Called after leaving the mainloop, e.g. after stop or quit."""
    self.logger.debug("on_quit")
    self.send_parallel_and_write(self.GAME_OVER)
    pygame.quit()


def tick(self):
    """
    Called repeatedly in the mainloop no matter if the Feedback is paused
    or not.
    """
    self.process_pygame_events()
    pygame.time.wait(10)
    # self.elapsed = self.clock.tick(self.FPS)
    pass


def pause_tick(self):
    """
    Called repeatedly in the mainloop if the Feedback is paused.
    """
    self.do_print("Pause", self.fontColor, self.size / 4)
    print
    'in pause'


def retrieve_word_list(self, username):
    words = self.db_query(
        'select word_index from word_list where username = ? order by sessionid desc limit 60')  # 180 originally


def add_to_word_list(self, word_index):
    self.db_query('insert into word_list(sessionid, username, word_index) values(?,?,?)',
                  (self.sessionid, self.username, word_index))


def type_name(self, message):
    pygame.event.clear()
    self.do_print(message)
    donezo = False
    self.username = ''

    shiftoggle = 0
    pygame.event.clear()
    answer = []
    korinput = korin.Coreano(answer)
    self.krtoggle = not self.krtoggle
    while not donezo:
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN):
                # msg += event.unicode
                km = pygame.key.name(event.key)
                if (km == 'left alt'):
                    # win32api.LoadKeyboardLayout('E0010412',1)
                    self.krtoggle = not self.krtoggle
                    korinput.flush()
                elif (km == 'right alt'):
                    # win32api.LoadKeyboardLayout('00000409',1)
                    self.krtoggle = not self.krtoggle
                elif (km == 'left shift'):
                    shiftoggle = 1

            if (event.type == pygame.KEYUP):
                _ = pygame.key.name(event.key)
                # the word is finished
                if (_ == 'return'):
                    donezo = True
                    time2 = time.time()

                # if the last entered letter shall be deleted
                elif ((_ == 'backspace') and (len(answer) > 0)):
                    answer.pop()
                    korinput.flush()
                # if the entered key is alphabetic it is part of the answer
                elif (_ == 'space'):
                    answer.append(' ')
                    korinput.flush()
                elif (_.isalpha()):
                    if not self.krtoggle:
                        answer.append(_)
                    else:
                        # answer.append(korin.korInput(_))
                        if shiftoggle:
                            korinput.takeChar(_.upper())
                        else:
                            korinput.takeChar(_)

                elif (_.isdigit()):
                    answer.append(_)

                elif (_ == 'left shift'):
                    shiftoggle = 0
                elif ((_ == '-') or (_ == ',')):
                    answer.append(_)
                self.do_print([message[0], message[1], "".join(unicode(item) for item in answer)])

    self.username = unicode(self.username.join(answer), "utf-8")
    print('your name is ' + self.username)
    self.krtoggle = not self.krtoggle


def ask_name(self):
    self.do_print(['Is this your first time with this experiment?', 'y for yes, n for no'], size=40, size_list=[40, 40])
    done = False
    message = []
    is_first_time = 0
    while not done:
        for event in pygame.event.get():
            if (event.type == pygame.KEYUP):
                # msg += event.unicode
                km = pygame.key.name(event.key)
                if (km == 'y'):
                    is_first_time = 1
                    done = True
                    message = ['Type two alphabets & birthday as your ID', ' ex) tj0131']
                elif (km == 'n'):
                    # win32api.LoadKeyboardLayout('00000409',1)
                    is_first_time = 0
                    done = True
                    message = ['Please type your ID', ' ']

    # message = ['Type two alphabets as your ID', ' ex) tj']

    self.type_name(message)
    if is_first_time:
        passed = False
        while not passed:
            result = self.db_query('select count(*) as count from session where username = ?', [self.username])
            if result.fetchone()[0] > 0:
                self.type_name(['username already exists.', 'try another one:'])
            else:
                passed = True

    self.db_query('update session set username = ? where sessionid = ?', (self.username, self.sessionid))


def play_tick(self):
    """
    Called repeatedly in the mainloop if the Feedback is not paused.
    """
    # subject_log = os.path.join(self.st_path, self.VP+'.db')
    # print 'subject log name:',os.path.isfile(subject_log)

    if self.typSes == 'Train_A' or self.typSes == 'Train_B' or self.typSes == 'Train_C':

        # "maybe not needed (duplicated in creat_db)"
        # sb_log = sqlite3.connect(subject_log)
        # cursor = sb_log.cursor()
        #
        # cursor.execute('SELECT max(sesion_id) in data')
        # last_session = cursor.fetchone()[0]

        self.c.execute('SELECT COUNT(*) from session WHERE username =?', (self.username,))
        data = self.c.fetchone()[0]

        "log is not empty"
        if data != 1:

            print
            'data log existed'

            "find last word index subject finished recently"
            self.c.execute('SELECT max(word_index) from word_list WHERE username=?', (self.username,))
            last_index = self.c.fetchone()[0]

            self.init_lection2(os.path.join(self.lection_path, self.Filenames))

            print
            "b4 dict size", len(self.dictionary)
            print
            "b4 dict indice", len(self.dict_indices)

            del self.dictionary[0:last_index]
            del self.dict_indices[0:last_index]
            self.bag = self.dict_indices[0:self.bag_size]  # Part B error Fix : moved code from init_lection2
            # print "after dict size",len(self.dictionary)
            # print "after dict indice",len(self.dict_indices)
            # print "dict ", self.dictionary[0:4]
            # print "dict+indice ", self.dict_indices[0:4]
            # print "dict+indice22 ", self.dict_indices.index(4)
            # print "self_bag_idx ", self.bag[0:5]
            # print "self_bag_size ", len(self.bag)
            # print "dict+indice2 ", self.dict_indices.index(1)#error


        else:
            print
            'data log empty'

            "log is empty (first experiment)"
            self.init_lection2(os.path.join(self.lection_path, self.Filenames))

            print
            'length(dic): ', len(self.dictionary), 'length(index): ', len(self.dict_indices)
            # print self.dictionary

        "pointer for math problem"
        tmp_index = (self.lession_num - 1) * 4

        self.training_oneShw()
        self.store_training()
        time.sleep(self.prePostTrain)
        answers = self.ask_maths_questions(self.maths_questions \
                                               [tmp_index:(tmp_index + 4)])
        self.store_maths_testing(answers)

        tmp_index += 4
        tmp_index = tmp_index % len(self.maths_questions)
        answer_array, test_dict = self.testing()
        self.store_testing(answer_array, test_dict)

    elif self.typSes == 'Test':

        final_dict = []

        self.init_lection2(os.path.join(self.lection_path, self.Filenames))

        last_session = self.sessionid - 1

        print
        "where are you"

        'get word indices learned from last 2 sessions'
        # self.c.execute('select word_index from word_list where username = ? order by timestamp desc limit 120 ',(self.username,))
        self.c.execute(
            'select wl.word_index from word_list wl where wl.username = ? and wl.sessionid IN (select ss.sessionid from session ss where ss.username = ? and ss.session_type != "Test" order by ss.sessionid desc limit 3 ) order by wl.timestamp desc',
            (self.username, self.username))
        final_indices = [int(record[0]) for record in self.c.fetchall()]
        self.dict_indices = final_indices

        for item in self.dict_indices:
            final_dict.append(self.dictionary[item - 1])
        answer_array, test_dict = self.testing()
        self.store_testing(answer_array, test_dict)

    self._running = False


def init_lection2(self, filename):
    """sets up the parameters which are used for each lection"""
    # here the word pairs are stored
    # finnish
    if self.finnish:
        self.dictionary = self.make_finnish_dictionary(filename)
    # chinese
    else:  # needs to read dictionary indices too in the third column of the text files
        [self.dictionary, self.dict_indices] = self.make_dictionary(filename)
    # create separate dictionary for distractors
    # reading absolute indices required
    # self.distrctDict = self.dictionary[self.distrInd[0]:self.distrInd[1]]
    #        if (len(self.dictionary) > self.max_dict_length): # for new paradigm this probably does'nt matter
    #            self.dictionary = self.dictionary[0:self.max_dict_length]

    # temp dictionary for reference....... 추가됨

    [self.dictionary_reference, self.dictionary_indices_reference] = self.make_dictionary(filename)

    # change the length of dictionary size
    self.array_correct = numpy.zeros((self.tot_words, self.nr_of_max_reps))
    # this stores the trial number
    self.array_trial = numpy.zeros((self.tot_words, self.nr_of_max_reps))

    # this is the array which stores the reaction times
    self.array_RT = numpy.zeros((self.tot_words, self.nr_of_max_reps))
    self.array_RT_1st_press = numpy.zeros((self.tot_words, self.nr_of_max_reps))

    self.list_correct = numpy.zeros(len(self.dictionary))

    # here is stored which pairs should be called
    self.bag_size = self.initial_bag_size
    if (self.bag_size > len(self.dictionary)):
        self.bag_size = len(self.dictionary)

    self.bag = self.dict_indices[0:self.bag_size]  # need to be replaced with the index of the word
    # probably extract into the dictionary itself

    # here is stored which is the next index of the dictionary
    # the bag can be filled with
    self.bag_filling_index = self.bag_size
    # self.sequence = list(numpy.arange(len(self.dictionary)))
    # self.ask_sequence = make_initial_ask_sequence()
    self.showed_sequence = []
    self.asked_sequence = []
    # self.index = 0
    # this is the index where the pair which has not been remembered should be inserted in the sequence
    # self.incorrect_index = len(self.dictionary)
    # self.answers = []


def handleEvents(self):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
        else:
            self.running = False


def update_bag(self, pair_to_remove):
    self.handleEvents()
    """removes the pair out of the bag and puts a new one in"""
    # NOTE: to remove the same element from both bad and bad_index
    #        1. reassign bag_index to bag, it'll be shuffled anyway
    #        2. search for the common element and remove based on host index

    self.learnt_words += 1
    print
    '+++++++++++++++++++++++++++++++++++++++++++++++++'
    print
    'Learnt Words: ' + str(self.learnt_words) + '/' + str(len(self.dict_indices))
    print
    '*************************************************'
    # Using method 1: Test if this works. But still distractors are not bein
    # accessed
    self.bag_index = []
    self.bag_index = self.bag[:]

    print
    'bag_(before update):', self.bag

    self.bag.remove(pair_to_remove)
    self.bag_index.remove(pair_to_remove)

    if self.learnable_word_count < self.learn_words - 10:

        known = True

        while (known):
            if (self.bag_filling_index < len(self.dictionary)):

                tmp = self.dictionary[self.bag_filling_index]
                self.send_parallel_and_write(self.NEW_PAIR, self.dict_indices[self.bag_filling_index])
                self.show_pair(tmp[0:2])

                self.do_print(['Did you already know this word?', '1. Yes    2. No'])

                time1 = time.time()
                time2 = 0
                answer = []
                done = False
                wait_for_1st_press = True

                while not done:
                    for event in pygame.event.get():
                        if (event.type == pygame.KEYUP):
                            if wait_for_1st_press:
                                time3 = time.time()
                                wait_for_1st_press = False
                            _ = pygame.key.name(event.key)

                            # if the last entered letter shall be deleted
                            if (_ == '1'):
                                print
                                'yeah'
                                answer.append('1. Yes')
                                done = True
                                known = True
                                self.known_pair_index.append(self.dict_indices[self.bag_filling_index])
                                print
                                'knwon word added, current index list is: ', self.known_pair_index

                            elif (_ == '2'):
                                print
                                'no'
                                answer.append('2. No')
                                done = True
                                known = False
                                self.add_to_word_list(self.dict_indices[self.bag_filling_index])
                                self.bag.append(self.dict_indices[
                                                    self.bag_filling_index])  # need to be replaced by self.dict_indices
                                self.bag_index.append(self.dict_indices[self.bag_filling_index])


                            elif ((_ == 'backspace') and (len(answer) > 0)):
                                answer.pop()
                            # if the entered key is alphabetic it is part of the answer
                            elif (_ == 'space'):
                                answer.append(' ')
                            elif (_.isalpha()):
                                answer.append(_)
                            elif ((_ == '-') or (_ == ',')):
                                answer.append(_)
                            self.do_print(['Did you already know this word?', "".join(answer)])

                # store which words have been shown
                self.showed_sequence.append(self.dict_indices[self.bag_filling_index])  # list to store shown words

                print
                "bag(updated): ", self.bag

                # no words have been asked here
                self.asked_sequence.append(-1)
                self.trial = self.trial + 1;
                self.bag_filling_index += 1

        # self.send_parallel_and_write(self.UPDATE, self.dict_indices[self.bag_filling_index],'Update')
        self.learnable_word_count += 1
        print
        "learnable_word_count ", self.learnable_word_count


# def check_knownpair(self):

def show_instruction(self):
    self.handleEvents()

    questions = []

    datei = open(self.instruction_path, 'r')
    for zeile in datei.readlines():
        zeile = " ".join(zeile.split("\n"))
        zeile = zeile.decode('utf-8')
        questions.append(zeile)
    datei.close()

    self.do_print([questions[0], questions[1],
                   questions[2], questions[3], questions[4]], self.fontColor,
                  size_list=[60, 60, 60, 60, 20], size=60, center=-1)
    self.wait_until_enter()


def training_oneShw(self):
    self.handleEvents()

    """ here all words are shown once and then only asked. upon wrong response, the
        correct word is then shown: thereby learning"""

    # training_done = False
    # exchange = False
    self.send_parallel_and_write(self.TRAINING, task='Training')
    self.do_print('Training')
    time.sleep(self.inter_time)
    time.sleep(self.prePostTrain)

    self.show_instruction()

    self.trial = 0

    # show all words in the bag
    self.bag_index = self.bag[:]
    current_bag_idx = 0
    # for pair_index in self.bag_index:# note indices of words are not yet included
    while current_bag_idx < 10:
        # NOTE: pair index is the absolute index of words

        # temp_ind = self.bag.index(pair_index)
        # NOTE: since dictionary is not changing with bag, the temp_ind
        #       refers to the first five elements only. This needs to be
        #       updated as well. Sol: use (n-1)*60+1 (remove the 1 for py)
        #       then abs word number from bag can be used.
        # tmp = self.dictionary[temp_ind]
        # tmp = self.dictionary[self.dict_indices.index(pair_index)]
        print
        "self.curr_bag_index is ", current_bag_idx
        print
        "self.bag_index is ", self.bag_index[current_bag_idx]
        print
        "self.dict+indice is ", self.dict_indices.index(self.bag_index[current_bag_idx])

        tmp = self.dictionary[self.dict_indices.index(self.bag_index[current_bag_idx])]
        # self.send_parallel_and_write(self.NEW_PAIR, pair_index, 'present')
        self.show_pair(tmp[0:2])

        print
        "bag index", self.bag_index
        print
        "tmp", tmp

        self.do_print(['Did you already know this word?', '1. Yes    2. No', 'please press 1. yes or 2. no'],
                      size_list=[None, None, 30])

        time1 = time.time()
        time2 = 0
        answer = []
        done = False
        wait_for_1st_press = True
        known = False

        while not done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP):
                    if wait_for_1st_press:
                        time3 = time.time()
                        wait_for_1st_press = False
                    _ = pygame.key.name(event.key)

                    # if the last entered letter shall be deleted
                    if (_ == '1'):
                        print
                        'yeah'
                        answer.append('1. Yes')
                        done = True
                        known = True

                    elif (_ == '2'):
                        print
                        'no'
                        answer.append('2. No')
                        done = True

                    elif ((_ == 'backspace') and (len(answer) > 0)):
                        answer.pop()
                    # if the entered key is alphabetic it is part of the answer
                    elif (_ == 'space'):
                        answer.append(' ')
                    elif ((_ == '-') or (_ == ',')):
                        answer.append(_)
                    self.do_print(['Did you already know this word?', "".join(answer), 'please press 1. yes or 2. no'],
                                  size_list=[None, None, 30])

        # store which words have been shown
        if (known == False):

            self.logger.debug("new pair is shown")
            pair_index = self.bag_index[current_bag_idx]
            self.send_parallel_and_write(self.NEW_PAIR, pair_index, 'present')  # should be sent earlier than this..
            self.add_to_word_list(pair_index)
            current_bag_idx += 1
            self.showed_sequence.append(pair_index)  # list to store shown words
            # no words have been asked here
            self.asked_sequence.append(-1)
            self.trial = self.trial + 1;

            print
            'pair_index', pair_index

        else:
            self.known_pair_index.append(self.bag_index[current_bag_idx])
            print
            'knwon word added, current index list is: ', self.known_pair_index
            self.bag_index.remove(self.bag_index[current_bag_idx])
            self.bag.remove(self.bag[current_bag_idx])

            if (self.bag_filling_index < len(self.dictionary)):
                self.bag.append(self.dict_indices[self.bag_filling_index])  # need to be replaced by self.dict_indices
                self.bag_index.append(self.dict_indices[self.bag_filling_index])
                self.bag_filling_index += 1

        # # store which words have been shown
        # self.showed_sequence.append(pair_index) #list to store shown words
        # # no words have been asked here
        # self.asked_sequence.append(-1)
        # self.trial = self.trial+1;
    # training without distractors i.e., until bag size is big enough without distractors
    print
    '+++++++++++++++++++++++++++++++++++++++++++++++++'
    print
    'Learnt Words: ' + str(self.learnt_words) + '/' + str(len(self.dict_indices))
    print
    '*************************************************'

    while (len(self.bag_index) > self.minBagSize):  # this maintains a constant bag size else #self.distractor_size):

        #            print 'self.array_trial= '+str(self.array_trial)
        #            print 'self.array_correct= '+str(self.array_correct)
        #            print 'self.bag_index= '+str(self.bag_index)

        numpy.random.shuffle(self.bag_index)
        rand_index = self.bag_index[0]
        #            print 'self.bag_index= '+str(self.bag_index)
        #            print 'self.bag= '+str(self.bag)
        if len(self.showed_sequence) < self.show_display_limit + 1:
            print
            'shown_seq= ' + str(self.showed_sequence)
        else:
            print
            'shown_seq= ' + str(self.showed_sequence[-1 * self.show_display_limit:])
        # check if the word is not repeated for at least a gap of distractor_size
        presentation = self.verify_gap(rand_index)
        print
        presentation
        # if N-x'th recall condition verified then finally ask the word
        if presentation:
            temp_index = self.bag.index(rand_index)
            self.present_word(temp_index)
        #            print 'BAG-SIZE is '+str(len(self.bag_index))
        # training with distractors i.e., when bag size has shrunk below minBagSize
        #            print '+++++++++++++++++++++++++++++++++++++++++++++++++'
        print
        'Learnt Words: ' + str(self.learnt_words) + '/' + str(len(self.dict_indices))
    #            print '*************************************************'
    print
    '************************************************************'
    print
    'DISTRACTOR SEQUENCE ENTERED :)'
    print
    '************************************************************'
    while (len(self.bag_index) > 0) and (len(self.bag_index) <= self.minBagSize):
        # read the distractors into a separate dictionary

        # then proceed like above
        # go through the bag number of times == length of the bag
        # or just put a if else clause
        for _ in range(len(self.bag_index)):
            numpy.random.shuffle(self.bag_index)
            rand_index = self.bag_index[0]  # call this pair or abs_index
            # check if the word is not repeated for at least a gap of distractor_size
            presentation = self.verify_gap(rand_index)
            if presentation:
                # distractor = False
                temp_index = self.bag.index(rand_index)
                self.present_word(temp_index)
                print
                'BAG-SIZE is ' + str(len(self.bag_index))
                break
        #            print 'self.bag_index= '+str(self.bag_index)
        #            print 'self.bag= '+str(self.bag)
        if len(self.showed_sequence) < self.show_display_limit + 1:
            print
            'show_seq= ' + str(self.showed_sequence)
        else:
            print
            'show_seq= ' + str(self.showed_sequence[-1 * self.show_display_limit:])
        #            print 'presentation is '+str(presentation)
        #            print '+++++++++++++++++++++++++++++++++++++++++++++++++'
        print
        'Learnt Words: ' + str(self.learnt_words) + '/' + str(len(self.dict_indices))
        #            print '*************************************************'

        if not (presentation):
            #                print 'Distractorss++++'
            # if none of the cycles above gave a positive presentation, then
            # use a distractor
            # show all distractor first: for every distractor loop only one
            if len(self.distr_show) < len(self.distractor_bag):
                rand_index = len(self.distr_show)
                self.logger.debug("distractor is asked")
                self.send_parallel_and_write(self.DISTRACTOR, \
                                             self.distractor_bag[rand_index], \
                                             'distractor')
                tmp = self.distrctDict[rand_index]
                self.show_pair(tmp[0:2])
                self.distr_show.append(self.distractor_bag[rand_index])
                self.trial = self.trial + 1

                # store in arrays
                self.showed_sequence.append(self.distractor_bag[rand_index])
                self.asked_sequence.append(self.distractor_bag[rand_index])

            else:
                rand_index = numpy.random.randint(0, len(self.distractor_bag))
                # numpy.random.shuffle(self.distractor_bag)
                # rand_index = self.distractor_bag[0]
                presentation = True
                # distractor = True

                # present the distractor (NOTE: distractors were not presented earlier)
                #    or should the distractors be presented along with the training bag
                self.logger.debug("distractor is asked")
                #                temp_ind =
                self.send_parallel_and_write(self.DISTRACTOR_ASK, \
                                             self.distractor_bag[rand_index], \
                                             'distractor_ask')
                _ = self.ask_answer(self.distrctDict[rand_index], \
                                    self.distractor_bag[rand_index])
                # '_' is a temporary variable
                self.trial = self.trial + 1

                # store in arrays
                self.showed_sequence.append(self.distractor_bag[rand_index])
                self.asked_sequence.append(self.distractor_bag[rand_index])
                # for distractors do we need to store the results???
                #                self.array_correct[self.distractor_bag[rand_index]][ind] = -1.0
                #                self.array_trial[self.bag[rand_index]][ind] =  self.trial
                print
                'BAG-SIZE is ' + str(len(self.bag_index))

        #            print '+++++++++++++++++++++++++++++++++++++++++++++++++'
        print
        'Learnt Words: ' + str(self.learnt_words) + '/' + str(len(self.dict_indices))


def present_word(self, rand_index):
    """ tests a given pair and stores the results in respective arrays"""
    # rand_index is temp_index from calling function: refers to the index of words in bag
    # Problem in this section [11.1.11,18.82]
    # NOTE: Also the naming of variable rand_index is confusing. Need to
    #       change.
    exchange = False
    word_index = self.bag[rand_index] - 1  # array_correct, trial, RT have
    # tot_words x max_rep size, hence
    # word_index refers to absolute word
    # index. Note: the inidices of the
    # above arrays are abs indices
    # already
    pair_index = self.bag[rand_index]

    self.logger.debug("test is asked")
    self.send_parallel_and_write(self.ASK_TRAIN, self.bag[rand_index], task='ask_train')

    answer = self.ask_answer(self.dictionary[self.dict_indices.index(pair_index)], self.bag[rand_index])
    # send markers
    ##MOVED MARKER SENDING TO ask_answer
    if answer[0]:
        # self.send_parallel_and_write(self.TRAINING_CORRECT, self.bag[rand_index])
        print
        self.TRAINING_CORRECT
    else:
        # self.send_parallel_and_write(self.TRAINING_INCORRECT, self.bag[rand_index])
        print
        self.TRAINING_INCORRECT

    self.trial = self.trial + 1;  # update the trial number
    # stores which pair has been asked NOTE: showed and asked sequence is the same
    # except for initial all show presentation
    self.showed_sequence.append(self.bag[rand_index])
    self.asked_sequence.append(self.bag[rand_index])
    # stores the reaction time
    # NOTE: 1.rand_index refers to
    ind = list(self.array_correct[word_index]).index(0.0)
    self.array_RT[word_index][ind] = answer[2]
    self.array_RT_1st_press[word_index][ind] = answer[3]

    # storing answer in array_correct and array_trial
    if answer[0]:
        # if the answer has been correct 1 is stored
        # NOTE:1.problem with passing rand_index since it is temporary
        #      2.also bag index cannot be used either since the bag contents
        #        can change
        #      3.hence best use an index that refers to word index in
        #        array_correct and array_trial
        self.array_correct[word_index][ind] = 1.0
        self.array_trial[word_index][ind] = self.trial
        # if the answer was often enough correct
        # the word is thrown out of the bag
        if (ind >= (self.nr_of_correct_reps - 1)):
            exchange = True
            for i in range(self.nr_of_correct_reps):
                if (self.array_correct[word_index][ind - i] == -1.0):
                    exchange = False

            if exchange:  # need to look into how this update works as well
                self.update_bag(self.bag[rand_index])

    else:
        # if the answer has not been correct -1 is stored
        self.array_correct[word_index][ind] = - 1.0
        self.array_trial[word_index][ind] = self.trial

    # if the word was asked for often enough it gets also thrown out of the bag
    # or if there's only one word left, the lesson is finished as well
    #
    if (not (exchange) and (ind == (self.nr_of_max_reps - 1))):
        self.update_bag(self.bag[rand_index])


def verify_gap(self, rand_index):
    """ returns 'True' if the word is not repeated for at least a gap of
        distractor_size"""
    value = list(cmp(rand_index, varble) for varble in self.showed_sequence \
        [len(self.showed_sequence) - self.distractor_size:])
    #        print 'value= '+str(value)
    #        decision_var = sum(value)
    #        temp = 1 in value
    if 0 in value:  # decision_var == -1 * self.distractor_size:
        return False
    else:
        return True


def show_pair(self, pair, first_seen=False):
    """
    shows one pair
    """
    # Fixation Part
    self.logger.debug("Fixation period started")  # what is this fixation period??
    ##UNNECESSARY self.send_parallel_and_write(self.FIXATION_PERIOD, task='show')
    print
    self.FIXATION_PERIOD
    # time.sleep(self.fixation_time)

    # Presentation Part
    self.logger.debug("Presentation period started")
    # self.send_parallel(self.PRESENTATION_PERIOD ) ##REMOVED
    print
    self.PRESENTATION_PERIOD
    # shows the sign and the word
    if first_seen:

        ##NO MORE ENTER

        self.do_print([pair[0], '', ""],
                      size_list=[None, None, 20], center=0)

        time.sleep(self.presentation_time)
        pygame.event.clear()  # only for presentationtime
        # self.wait_until_enter()

        self.do_print([pair[1], '', ""],
                      size_list=[None, None, 20], center=0)
        time.sleep(self.presentation_time)
        pygame.event.clear()  # only for presentationtime
        # self.wait_until_enter2()

        # show both words together in relation
    else:
        self.do_print([" - ".join(pair), '', ""],
                      size_list=[None, None, 20], center=0)
        # self.wait_until_enter2()
        time.sleep(self.presentation_time)
        pygame.event.clear()  # only for presentationtime

    # Inter Part
    self.logger.debug("Inter period started")
    # self.send_parallel(self.INTER_PERIOD) ##UNNECESSARY
    self.do_print('')
    # time.sleep(self.inter_time)


def ask_test(self, pair, index):
    """
    shows one pair
    """
    # Fixation Part
    self.logger.debug("Fixation period of test started")
    ##UNNECESSARY self.send_parallel(self.FIXATION_PERIOD_TEST)
    time.sleep(self.fixation_time_test)

    # Presentation Part
    self.logger.debug("Presentation period of test started")
    ##UNNECESSARY self.send_parallel(self.PRESENTATION_PERIOD_TEST)
    test_list = ['TEST:', 'What is', pair[0]]
    answer_list = [pair[1]]
    # answer_list = [pair[2]]
    next_4_elems = []
    for _ in range(4):
        elem_index = random.randint(0, numpy.shape(self.dictionary)[0] - 1)
        while ((elem_index in next_4_elems) or (elem_index == index)):
            elem_index = random.randint(0, numpy.shape(self.dictionary)[0] - 1)
        next_4_elems.append(elem_index)
        answer_list.append(self.dictionary[elem_index][1])
        # answer_list.append(self.dictionary[elem_index][2])
    random.shuffle(answer_list)

    showed_list = []
    showed_list.extend(answer_list)
    showed_list.insert(0, '')
    showed_list.insert(0, ' '.join([pair[0], ':']))
    showed_list.append('no idea')
    self.do_print(test_list, size=40)
    time.sleep(self.fixation_time_test)
    self.do_print(showed_list, size=30)
    # time.sleep(self.fixation_presentation_test_max)
    done = False
    time1 = time.time()
    time2 = 0
    answer = 0
    while not done:
        for event in pygame.event.get():
            if ((event.type == pygame.KEYDOWN) or (event.type == pygame.KEYUP)):
                _ = pygame.key.name(event.key)
                if (_ in ['1', '2', '3', '4', '5', '6']):
                    answer = int(_)
                #                   if ((answer == 6) or (answer_list[answer - 1] != pair[2])):
                if ((answer == 6) or (answer_list[answer - 1] != pair[1])):
                    correct = False
                else:
                    correct = True
                done = True
                time2 = time.time()

    self.do_print('')
    time.sleep(self.inter_time_test)

    # This is the direct feedback for the user
    if correct:
        self.do_print('Correct', size=30)
        time.sleep(self.user_feedback_time)
    else:
        #            self.do_print(['The answer has not been correct:', '', pair[0], 'means', pair[2]], size=30)
        self.do_print(['Wrong :', '', \
                       " - ".join(pair), " "],
                      size_list=[30, 30, None, 20], center=2)
        # self.wait_until_enter()
        time.sleep(self.presentation_time)
    # time.sleep(self.user_feedback_time)

    # Inter Part
    self.logger.debug("Inter period of test started")
    ##UNNECESSARY self.send_parallel(self.INTER_PERIOD_TEST)
    self.do_print('')
    time.sleep(self.inter_time_test)

    # returns whether the answer has been correct and at which position the
    # correct word has been written
    position = 1 + answer_list.index(pair[1])
    reaction_time = time2 - time1
    return [correct, position, reaction_time]


def ask_answer(self, pair, index, give_feedback=True):
    """asks for the answer"""

    # replace 171/172 to to respective codes (161/162 for training, 151/152 for testing parts)
    if self.part == 1 or self.part == 2 or self.part == 4 or self.part == 5:
        correct_code = self.TRAINING_CORRECT
        incorrect_code = self.TRAINING_INCORRECT
    else:
        correct_code = self.TESTING_CORRECT
        incorrect_code = self.TESTING_INCORRECT

    # Fixation Part
    self.logger.debug("Fixation period of test started")
    # self.send_parallel(self.FIXATION_PERIOD_TEST) ##UNNECESSARY
    # time.sleep(self.fixation_time_test)

    # Presentation Part
    self.logger.debug("Presentation period of test started")
    # self.send_parallel(self.PRESENTATION_PERIOD_TEST) ##UNNECESSARY
    print
    "pair is ", pair
    showed_item = " ".join(['What is ', pair[0], '?'])

    self.do_print(showed_item)
    time1 = time.time()
    time2 = 0
    time3 = 0
    done = False
    wait_for_1st_press = True
    answer = []
    msg = ''
    korinput = korin.Coreano(answer)

    self.add_event(index, self.past_test_start)
    last_event_key = self.c.lastrowid

    # self.krtoggle = 0
    shiftoggle = 0
    # make sure that question is really seen
    time.sleep(self.time_b4_enter_press)
    pygame.event.clear()
    RESETEVENT = pygame.USEREVENT + 1
    while not done:
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN):
                # msg += event.unicode
                # print(msg," and ", answer, " input : ", event.unicode)
                # print pygame.key.name(event.key)
                km = pygame.key.name(event.key)
                if (km == 'left alt'):
                    # win32api.LoadKeyboardLayout('E0010412',1)
                    self.krtoggle = not self.krtoggle
                    korinput.flush()
                elif (km == 'right alt'):
                    # win32api.LoadKeyboardLayout('00000409',1)
                    self.krtoggle = self.krtoggle

                if (pygame.key.name(event.key) == 'left shift'):
                    shiftoggle = 1
                    print
                    "enter shift"

            if event.type == RESETEVENT:  # called 0.1 seconds after pressing shift
                shiftoggle = 0
                pygame.time.set_timer(RESETEVENT, 0)
                print
                "out of shift"

            if (event.type == pygame.KEYUP):
                if wait_for_1st_press:
                    time3 = time.time()
                    self.send_parallel_and_write_aq(self.FIRST_PRESS, index, last_event_key)
                    wait_for_1st_press = False
                _ = pygame.key.name(event.key)
                # the word is finished
                if (_ == 'return'):
                    done = True
                    time2 = time.time()
                    # self.send_parallel(self.LAST_PRESS)
                    if ("".join(answer) == pair[1]):
                        correct = True
                        # self.send_parallel(self.ANSWER_CORRECT)
                        # self.send_parallel_and_write(correct_code, index)
                        self.send_parallel_and_write_aq(correct_code, index, last_event_key)
                    ##HACK FOR fast iteration
                    elif ("".join(answer) == 'qp'):
                        correct = True
                        # self.send_parallel(self.ANSWER_CORRECT)
                        # self.send_parallel_and_write(correct_code, index)
                        self.send_parallel_and_write_aq(correct_code, index, last_event_key)
                    else:
                        correct = False
                        # self.send_parallel(self.ANSWER_INCORRECT)
                        # self.send_parallel_and_write(incorrect_code, index)
                        self.send_parallel_and_write_aq(incorrect_code, index, last_event_key)
                        print
                        self.ANSWER_INCORRECT
                    self.add_response(last_event_key, index, "".join(answer), correct)

                # if the last entered letter shall be deleted
                elif ((_ == 'backspace') and (len(answer) > 0)):
                    answer.pop()
                    korinput.flush()
                # if the entered key is alphabetic it is part of the answer
                elif (_ == 'space'):
                    answer.append(' ')
                    korinput.flush()

                elif (_.isdigit()):
                    answer.append(_)

                elif (_.isalpha()):
                    if not self.krtoggle:
                        answer.append(_)
                    else:
                        # answer.append(korin.korInput(_))
                        if shiftoggle:
                            korinput.takeChar(_.upper())
                        else:
                            korinput.takeChar(_)
                elif (_ == 'left shift'):
                    # shiftoggle = 0
                    pygame.time.set_timer(RESETEVENT, 100)  # to improve user experience in using shift
                elif ((_ == '-') or (_ == ',')):
                    answer.append(_)
                    # self.do_print([showed_item, "".join(answer)])
                self.do_print([showed_item, "".join(unicode(item) for item in answer)])

    # This is the direct feedback for the user
    if give_feedback:
        if correct:
            self.do_print('Correct', color=(50, 155, 50), \
                          size=30)
            time.sleep(self.user_feedback_time)
        else:
            self.do_print(['Wrong', \
                           '', " - ".join(pair), ""], \
                          color=(215, 50, 50), \
                          size_list=[30, 30, None, 20], center=2)
            # self.wait_until_enter2()
            time.sleep(self.presentation_time)
            self.logger.debug("Inter period of test started")
            ##UNNECESSARY self.send_parallel(self.INTER_PERIOD_TEST)
            self.do_print('')
            time.sleep(self.inter_time_test)

    self.do_print('')

    position = 1
    reaction_time = time2 - time1
    reaction_time_1st_press = time3 - time1
    return [correct, position, reaction_time, reaction_time_1st_press]


def ask_maths_questions(self, questions):
    """asks some mathematical questions"""
    # Fixation Part
    self.logger.debug("Period of math test started")
    self.send_parallel_and_write(self.MATH_TEST, task='maths')
    self.do_print('Maths Testing')
    time.sleep(self.inter_time_test)

    m_answers = []

    for q in questions:

        pair = q.split('\t')
        # Fixation Part
        # self.logger.debug("Fixation period of test started")
        ##UNNECESSARY self.send_parallel(self.FIXATION_PERIOD_TEST)
        time.sleep(self.fixation_time_test)

        # Presentation Part
        # self.logger.debug("Presentation period of test started")
        ##UNNECESSARY self.send_parallel(self.PRESENTATION_PERIOD_TEST)
        showed_item = " ".join(['What is ', pair[0], '?'])
        self.do_print(showed_item)
        done = False
        answer = []
        correct = False
        # make sure that question is really seen
        time.sleep(self.time_b4_enter_press)
        pygame.event.clear()
        while not done:
            for event in pygame.event.get():
                if (event.type == pygame.KEYUP):
                    _ = pygame.key.name(event.key)
                    #                       # the word is finished
                    if (_ == 'return'):
                        done = True
                        if ("".join(answer) == pair[1].split(' ')[0]):
                            correct = True
                            self.send_parallel(self.ANSWER_CORRECT)
                        else:
                            correct = False
                            self.send_parallel(self.ANSWER_INCORRECT)

                    #                       # if the last entered letter shall be deleted
                    elif ((_ == 'backspace') and (len(answer) > 0)):
                        answer.pop()
                    #                       # if the entered key is alphabetic it is part of the answer
                    #                       #elif(_.isalpha()):
                    else:
                        answer.append(_)
                    self.do_print([showed_item, "".join(answer)])
        self.do_print('')
        m_answers.append(correct)
    return m_answers


def testing(self):
    """tests the whole dictionary"""
    # makes a deep copy of dictionary and shuffles it
    test_dictionary = []
    self.past_test_start = 1

    # if self.known_pair_index !=[]:
    #     for item in self.known_pair_index:
    #
    #         self.dict_indices.remove(item)
    #         # self.dictionary[]
    #     if len(self.dict_indices) !=60:
    #         for count in (len(self.dict_indices)-60):
    #             del self.dict_indices[-1]
    #     temp_indices = range(len(self.dict_indices))
    #
    #
    #
    # else:
    # temp_indices = range(len(self.dict_indices))
    # if len(self.dict_indices) !=60:
    #     for count in (len(self.dict_indices)-60):
    #         del self.dict_indices[-1]
    if self.typSes != 'Test':
        self.c.execute('SELECT word_index from word_list where sessionid = ?', (self.sessionid,))
        test_indices = [int(record[0]) for record in self.c.fetchall()]

        temp_indices = range(len(test_indices))
        print
        "dict_ref : ", len(self.dictionary_reference)

        for item in test_indices:
            test_dictionary.append(self.dictionary_reference[item - 1])
        print
        "session test? ", len(test_indices)

        self.dict_indices = test_indices

    else:
        print
        "dict_indices? ", len(self.dict_indices)
        test_indices = self.dict_indices
        temp_indices = range(len(test_indices))
        test_dictionary.extend(self.dict_indices)

    print
    len(test_indices)
    print
    test_dictionary
    print
    "dict size", len(self.dictionary)

    #        test_dictionary=[]
    random.shuffle(temp_indices)
    # check that the first item which is asked is not the last shown one
    #        if (self.part != 3):
    #            while (test_dictionary[0] == self.dictionary[self.asked_sequence[len(self.asked_sequence) - 1]]):
    #                random.shuffle(test_dictionary)
    print
    'shuffled index ', temp_indices

    # here the answers are stored: correct? reaction time? reaction_time_1st_press
    answer_array = numpy.zeros((len(test_dictionary), 3))
    test_order = []
    index = 0
    # every pair is tested
    self.send_parallel_and_write(self.TESTING, task='testing')
    self.do_print('Testing')
    time.sleep(self.inter_time_test)
    for i in temp_indices:
        if self.typSes == 'Test':
            pair = self.dictionary_reference[test_dictionary[i] - 1]  # originally test_dictionary[i]
        else:
            pair = test_dictionary[i]  # originally test_dictionary[i]
            "************************************************"
        # self.send_parallel_and_write()
        self.send_parallel_and_write(self.TEST, self.dict_indices[i], task='test')
        "************************************************"
        print
        str(temp_indices.index(i) + 1), ': '
        answer = self.ask_answer(pair, self.dict_indices[i], False)
        # if answer[0]:
        #     self.send_parallel_and_write(self.TESTING_CORRECT, \
        #                                  self.dict_indices[i])
        #     print ""
        #
        # else:
        #     self.send_parallel_and_write(self.TESTING_INCORRECT, \
        #                                  self.dict_indices[i])
        #     print ""
        test_order.append(self.dict_indices[i])
        answer_array[index][0] = answer[0]
        answer_array[index][1] = answer[2]
        answer_array[index][2] = answer[3]
        index += 1
    # here the percentage of the correct answers is calculated and
    # presented to the user as some feedback
    pc = list(answer_array.T[0]).count(True) * 100 / len(test_dictionary)
    self.do_print(' '.join(['Your answers were', str(pc), '% correct']), \
                  size=30)
    time.sleep(self.final_result_time)
    return answer_array, test_order


def wait_until_enter(self):
    """ waits until the enter button has been pressed """
    done = False
    # make sure that word is really seen
    time.sleep(self.time_b4_enter_press)
    pygame.event.clear()
    while not done:
        for event in pygame.event.get():
            if (event.type == pygame.KEYUP):
                _ = pygame.key.name(event.key)
                if (_ == 'return'):
                    done = True


def wait_until_enter2(self):
    """ waits until the enter button has been pressed """
    done = False
    # make sure that word is really seen
    time.sleep(self.time_b4_enter_press)
    pygame.event.clear()
    while not done:
        for event in pygame.event.get():
            if (event.type == pygame.KEYUP):
                _ = pygame.key.name(event.key)
                if (_ == 'return'):
                    # self.send_parallel_and_write(self.POSSIBLE_LEARNING, \
                    #                         task='possible learnin period')
                    # print self.POSSIBLE_LEARNING
                    done = True


def on_control_event(self, data):
    self.logger.debug("on_control_event: %s" % str(data))
    self.f = data["data"][- 1]
    print
    data


def on_interaction_event(self, data):
    # this one is equivalent to:
    # self.myVariable = self._someVariable
    self.myVariable = data.get("someVariable")
    print
    self.myVariable
    print
    data


def init_pygame(self):
    """
    Sets up pygame and the screen and the clock.
    """
    pygame.init()
    pygame.display.set_caption('Vocabulary Developer Feedback')

    #       so that the pygame window opens and fits exactly on the display screen
    if self.hostname == 'Rithwik-PC':
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(800) + "," + str(0)
    else:
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(0) + "," + str(20)
    if self.fullscreen:
        self.screen = pygame.display.set_mode((self.screenWidth, \
                                               self.screenHeight), \
                                              pygame.FULLSCREEN)
    else:
        self.screen = pygame.display.set_mode((self.screenWidth, \
                                               self.screenHeight), \
                                              pygame.RESIZABLE)
    self.w = self.screen.get_width()
    self.h = self.screen.get_height()
    self.clock = pygame.time.Clock()


def init_graphics(self):
    """
    Initialize the surfaces and fonts depending on the screen size.
    """
    self.screen = pygame.display.get_surface()
    (self.screenWidth, self.screenHeight) = (self.screen.get_width(), \
                                             self.screen.get_height())
    self.size = min(self.screen.get_height(), self.screen.get_width())
    # self.borderWidth = int(self.size*self.borderWidthRatio/2)

    # background
    self.background = pygame.Surface((self.screen.get_width(), \
                                      self.screen.get_height()))
    self.backgroundRect = self.background.get_rect \
        (center=self.screen.get_rect().center)
    self.background.fill(self.backgroundColor)


def do_print(self, text, color=None, size=None, superimpose=False,
             size_list=None, center=-1):
    """
    Print the given text in the given color and size on the screen.
    If text is a list, multiple items will be used, one for each list entry.
    """

    u_type = type(u'\u4e36')

    if not color:
        color = self.fontColor
    if not size:
        size = self.size / 10
    font = pygame.font.Font(''.join([self.lection_path, 'Cyberbit.ttf']), \
                            size)

    if not superimpose:
        self.draw_init()

    if type(text) is list:
        height = pygame.font.Font.get_linesize(font)
        top = -(2 * len(text) - 1) * height / 2
        for t in range(len(text)):
            if ((size_list != None) and (size_list[t] != None)):
                size = size_list[t]
            else:
                size = self.size / 10
            if (t != 0):
                color = self.fontColor
            font = pygame.font.Font(''.join([self.lection_path, \
                                             'Cyberbit.ttf']), size)
            tt = text[t]
            if (type(tt) != u_type):
                tt.decode("utf-8")
            surface = font.render(tt, 1, color)
            if (t == center):
                self.screen.blit(surface, surface.get_rect \
                    (center=self.screen.get_rect().center))
            else:
                self.screen.blit(surface, \
                                 surface.get_rect(midtop=(self.screenWidth / 2, \
                                                          self.screenHeight / 2 + top + t * 2 * height)))

    else:
        if (type(text) != u_type):
            text.decode("utf-8")
        surface = font.render(text, 1, color)
        self.screen.blit(surface, surface.get_rect \
            (center=self.screen.get_rect().center))
    pygame.display.update()


def draw_init(self):
    """
    Draws the initial screen.
    """
    self.screen.blit(self.background, self.backgroundRect)
    # self.screen.blit(self.border, self.borderRect)
    # self.screen.blit(self.inner, self.innerRect)


def process_pygame_events(self):
    """
    Process the the pygame event queue and react on VIDEORESIZE.
    """
    for event in pygame.event.get():
        if event.type == pygame.VIDEORESIZE:
            self.resized = True
            self.size_old = self.size
            h = min(event.w, event.h)
            self.screen = pygame.display.set_mode((event.w, h), \
                                                  pygame.RESIZABLE)
            self.init_graphics()
        elif event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            step = 0
            if event.unicode == u"a":
                step = -0.1
            elif event.unicode == u"d":
                step = 0.1
            self.f += step
            if self.f < -1: self.f = -1
            if self.f > 1: self.f = 1


def welcome(self):
    """shows welcome screen in beginning """

    self.logger.debug("Welcome started")
    ##UNNECESSARY self.send_parallel(777)
    ##UNNECESSARY self.send_parallel_and_write(666)

    self.do_print(["Welcome to the", "German Vocabulary Developer", '',
                   "Press enter to continue"],
                  size_list=[None, None, None, 20], center=1)
    self.wait_until_enter()
    # time.sleep(self.presentation_time)

    # Inter Part - 250ms after finger tap
    self.logger.debug("Inter period started")
    ##UNNECESSARY self.send_parallel(self.INTER_PERIOD)
    self.do_print('')
    time.sleep(self.inter_time)


def make_maths_questions(self, file):
    """ makes maths_questions from a given file"""
    questions = []
    datei = open(file, 'r')
    for zeile in datei.readlines():
        # get rid off \n at end of line
        zeile = " ".join(zeile.split("\n"))
        #            zeile = " ".join(zeile.split(";"))
        #            zeile = zeile.split("\t")
        #            zeile[0] = ncr_to_python(zeile[0])
        #            zeile[1] = "".join(zeile[1].split(" "))
        questions.append(zeile)
    datei.close()
    return questions


###THIS ONE IS NOT USED
#     def make_dictionary(self, file):
#         """ makes a dictionary from a given file and also reads the indices"""
#         dictionary = []
#         indices = []
#         datei = open(file,'r')
#         for zeile in datei.readlines():
#             # get rid off \n at end of line
#             zeile = " ".join(zeile.split("\n"))
#             zeile = " ".join(zeile.split(";"))
#             zeile = zeile.split("\t")
#             #print zeile[0]
#             zeile[0] = ncr_to_python(zeile[0])
#             #zeile[1] = "".join(zeile[1].split(" "))
#             zeile[1] = zeile[1].rstrip()
#             dictionary.append(zeile[0:2])
#             indices.append(int(zeile[2]))
# #            zeile[2] = "".join(zeile[2].split(" "))
#
#         datei.close()
#         result = [dictionary, indices]
#         return result
####THIS ONE IS USED INSTEAD
def make_dictionary(self, file):
    """ makes a dictionary from a given file and also reads the indices"""
    dictionary = []
    indices = []

    questions = []
    datei = open(file, 'r')
    for zeile in datei.readlines():
        zeile = " ".join(zeile.split("\n"))
        zeile = zeile.split("\t")
        zeile[0] = zeile[0].decode('utf-8')
        zeile[1] = zeile[1].decode('utf-8')
        dictionary.append(zeile[0:2])
        indices.append(int(zeile[2]))
    datei.close()

    result = [dictionary, indices]
    return result


def make_initial_ask_sequence(self):
    """makes the initial asking sequence"""
    seq = [-1, 0, -1]
    for i in range(len(self.dictionary)):
        seq[i + 1] = i
        i = i + 1
    return seq


# db declaration

def create_db(self):
    # create table
    self.c.execute('''CREATE TABLE IF NOT EXISTS session
            (sessionid integer primary key, username text, session_type text, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS raw_log
            (log_key integer primary key, sessionid integer, trigger integer, abs integer, task text, message text, event_key integer, 
            timestamp DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS stored_test
            (log_key integer primary key, sessionid integer, category text, data text, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS stored_math
            (log_key integer primary key, sessionid integer, category text, data text, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS stored_train
            (log_key integer primary key, sessionid integer, category text, data text, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS word_list
            (sessionid integer, username text, word_index integer, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS ask_event
                   (event_key integer primary key, sessionid integer, username text, word_index integer, ask_or_test text, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    self.c.execute('''CREATE TABLE IF NOT EXISTS user_answer
                   (event_key integer, sessionid integer, word_index integer, username text, user_response text, rightorwrong integer, timestamp DATETIME DEFAULT (datetime('now','localtime')))''')

    # introducting event concept for tests:
    # define a single test question as an event:
    # user answers, questions(word info), results, etc. will all be linked by a single event key
    # a separate table (ask_event) will be formed with the following:
    # event_key, user_key, sessionid, ask/test, word_idx, timestamp(at creation)

    # as an event consists of multiple actions across different time points,
    # time data aren't important except for tracking purposes

    # in addition, a table for user answers, right/wrong will be created as well.

    self.c.execute('INSERT INTO session (username) VALUES ("")'
                   )
    self.conn.commit()

    self.sessionid = self.c.lastrowid
    print
    "sessionid is " + str(self.sessionid)


def add_event(self, word_index=None, ask_or_test=None):
    self.db_query('insert into ask_event(sessionid, username, word_index, ask_or_test) values(?,?,?,?)',
                  (self.sessionid, self.username, word_index, ask_or_test))


def add_response(self, last_event_key, word_index=None, user_response=None, rightorwrong=None):
    self.db_query(
        'insert into user_answer(event_key, sessionid, username, word_index, user_response, rightorwrong) values(?,?,?,?,?,?)',
        (last_event_key, self.sessionid, self.username, word_index, user_response, rightorwrong))


def db_query(self, querytext, queryparam):
    self.conn.row_factory = sqlite3.Row
    self.c.execute(querytext,
                   queryparam)
    self.conn.commit()
    return self.c


# writedb
def send_parallel_and_db(self, code, trigger, abs_index=None, ):
    # nevermind... not used.
    self.send_parallel(trigger)


# trigger, abs_index are integers, task is a string
def send_parallel_and_write(self, trigger, abs_index=None, task=None, message=None):
    #        self.send_parallel(trigger)
    ctypes.windll.inpout32.Out32(self.pport_address, trigger)
    time.sleep(0.005)
    ctypes.windll.inpout32.Out32(self.pport_address, 0)
    # logfile = open(self.store_logfile,'a')
    #
    # t = time.localtime()
    # logfile.write(''.join([str(list(t)), '\t']))
    # logfile.write(''.join([str(trigger), '\t']))
    #
    # if(abs_index != None):
    #     logfile.write(''.join(['index:', str(abs_index), '\t']))
    # if(task != None):
    #     logfile.write(''.join(['task:', task, '\t']))
    #
    # logfile.write('\n')
    # logfile.close()

    self.c.execute('INSERT INTO raw_log (sessionid, trigger, abs, task, message) VALUES (?,?,?,?,?)',
                   (self.sessionid, trigger, abs_index, task, message))
    self.conn.commit()


def send_parallel_and_write_aq(self, trigger, abs_index=None, event_key=None):
    # self.send_parallel(trigger)
    ctypes.windll.inpout32.Out32(self.pport_address, trigger)
    time.sleep(0.005)
    ctypes.windll.inpout32.Out32(self.pport_address, 0)

    self.c.execute('INSERT INTO raw_log (sessionid, trigger, abs, event_key) VALUES (?,?,?,?)',
                   (self.sessionid, trigger, abs_index, event_key))
    self.conn.commit()


def store_training(self):
    """stores the training data(sequence, array_correct, array_RT)"""

    for _ in self.showed_sequence:
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'showed_sequence', str(_)))

    for _ in self.asked_sequence:
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'asked_sequence', str(_)))

    for i in range(numpy.shape(self.array_correct)[0]):
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'array_correct', str(list(self.array_correct[i]))))

    for i in range(numpy.shape(self.array_RT)[0]):
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'array_RT', str(list(self.array_RT[i]))))

    for i in range(numpy.shape(self.array_RT_1st_press)[0]):
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'array_RT_1st_press', str(list(self.array_RT_1st_press[i]))))

    for i in range(numpy.shape(self.array_trial)[0]):
        self.c.execute('INSERT INTO stored_train (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'array_trial', str(list(self.array_trial[i]))))

    self.conn.commit()


def store_testing(self, answer_array, test_dict):
    """stores the testing data"""

    for _ in test_dict:
        self.c.execute('INSERT INTO stored_test (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'test_dict', str(_)))
    for i in range(numpy.shape(answer_array)[0]):
        self.c.execute('INSERT INTO stored_test (sessionid, category, data) VALUES (?,?,?)',
                       (self.sessionid, 'answerarr', str(list(answer_array[i]))))
    self.conn.commit()


def store_maths_testing(self, answers):
    """stores the maths testing"""

    for _ in answers:
        self.c.execute('INSERT INTO stored_math (sessionid, data) VALUES (?,?)',
                       (self.sessionid, str(_)))
    self.conn.commit()


if __name__ == '__main__':
    vc = VocabularyDeveloperFeedback(None)
    vc.on_init()
    vc.on_play()
