from psychopy import gui
from psychopy import visual, core, event, logging, parallel, monitors
import random, glob, os, numpy, time, pylab
import pyxid
import datetime
import ctypes


class color_pretest():

    def __init__(self):

        # Markers sent to EEG machine
        self.run_start = 200
        self.run_end = 202

        self.black_t = 10
        self.black_nt = 11
        self.purple_t = 20
        self.purple_nt = 21
        self.blue_t = 30
        self.blue_nt = 31

        self.userInp_start = 90
        self.userInp_end = 91

        self.runscreen = 111
        self.crossscreen = 112
        self.endscreen = 113

        # Define variables (numbers currently used are for 1 run)
        self.max_run = 9  # 9
        self.symbolsPerRun = 240  # 240
        self.targetsPerRun = 60  # 60
        self.non_targetsPerRun = 180  # 180

        self.color = ['black', 'purple', 'blue']
        self.keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'return', 'backspace']
        self.bgcolor = (-0.5, -0.5, -0.5)
        self.fontName = 'Calibri'

        # Duration of stimuli on screen 
        # (symbol: cards)
        # (gap: interval between to cards being shown on screen)
        # (target: the target card which shown at beginning of each run that subject supposed to count)
        # (cross: screen center where subject supposed focus at)
        # (step: threshold to reset the timer)
        self.dur_symbol = 0.25  # 0.4 (400ms)
        self.dur_gap = 0.1  # 0.1 (100ms)
        self.dur_target = 2.0
        self.dur_cross = 2.0
        self.step = self.dur_symbol + self.dur_gap

        # Position coordinates(psychopy)
        # (range [-1,1], center [0,0])
        self.pos_title = (0, 0.7)
        self.pos_symbol = (0, 0)
        self.pos_input = (0, -0.2)
        self.pos_errorMsg = (0, 0.55)  # (when subject press enter without inputting a number) 

        # size
        self.cardSize = (0.3, 0.7)
        self.titleSize = 0.13
        self.errorSize = 0.06

        # initialize variables 
        self.process_priority = 'realtime'  # 'high' or 'realtime'
        self.flipTimes = []
        self.userAnswer = []
        self.run = 1
        self.divider = '\\'

        self.pport_address = 0xDFF8
        monitor = monitors.Monitor('testMonitor')
        self.win_size = monitor.getSizePix()
        self.win_size = (1920, 1080)

#        self.devices = pyxid.get_xid_devices()
        # self.stim = self.devices[-1]
        self.clock_ts = core.Clock()


    def main(self):
        #        self.pp.setData(self.test_start)

        self.loadCard()
        self.plyrInfo()
        self.createSeq()
        self.prepareLog()
        self.prepareTimestamp()
        self.initWindow()
        self.createStim()
        self.intro()

        while self.run <= self.max_run:
            self.runScreen()
            self.crossScreen()
            self.sequence()
            self.userInp()
            self.run += 1

        self.endScreen()
        self.addLog()
        self.plotframe()

    def prepareLog(self):

        # get path of current script
        self.Path = os.path.dirname(globals()["__file__"])

        # get log file path
        logpath = os.path.join(self.Path, 'log_pretest')

        # get current time (day-month-year_HourMinute)
        logtime = time.strftime('%d-%m-%Y_%Hh%Mm', time.localtime(time.time()))

        # define name of files created by this script(including log text file and frame report figure)
        self.filname = logtime + '_' + '(P)' + self.playerID

        # define log file name(txt format)
        self.Log = os.path.join(logpath, self.filname + '.txt')

        # open log file
        Log = open(self.Log, 'w')

        # write contents
        Log.write('Player info: ' + str(self.playerInfo) + '\n\n')
        Log.write('Targets: ' + str(self.targets) + '\n')

        for run in range(self.max_run):
            Log.write('Run ' + str(run + 1) + ': ')
            for symbol in range(self.symbolsPerRun):
                if symbol != self.symbolsPerRun - 1:
                    Log.write('(' + self.seqPerRun[run][symbol] + ', ' +
                              str(self.NumSeqperRun[run][symbol]) + ' ), ')
                else:
                    Log.write('(' + self.seqPerRun[run][symbol] + ', ' +
                              str(self.NumSeqperRun[run][symbol]) + ' )')
            Log.write('\n\n\n')

        # close log file
        Log.close()

    def prepareTimestamp(self):
        logpath = os.path.join(self.Path, 'log_pretest\\')
        logtime = time.strftime('%d-%m-%Y_%Hh%Mm', time.localtime(time.time()))
        self.timestamp_File = os.path.join(logpath,
                                           logtime + '_' + self.playerID + '_'  + 'Timestamp' + '.txt')

    def plyrInfo(self):

        # define GUI window for inputting subject information 
        # (position coordinate is depending on screen resolution, upper left corner [0,0])
        infoP = gui.Dlg(title="Lie detection experiment",
                        pos=((self.win_size[0] / 2) - 140, (self.win_size[1] / 2) - 100))

        # specify input field
        infoP.addText('Player Info', color='Purple')
        infoP.addField('Player Code:', 'VPxx')
        infoP.addText("'xx' is consist of any 2 alphabets, eg. VPyy", color='grey')
        infoP.addField('Gender:', choices=["Male", "Female"])
        infoP.addField('Age:')
        infoP.addField('Email address:', 'xxx@korea.ac.kr')
        infoP.addField('Handphone:', '010-xxxx-xxxx')

        # show GUI window to subject
        infoP.show()

        # button click for 'OK' and 'Cancel' 
        if infoP.OK:
            # if 'OK' is clicked
            # store data from subject to playerInfo
            self.playerInfo = infoP.data

            # reformat data
            for i in range(len(self.playerInfo)):
                self.playerInfo[i] = str(self.playerInfo[i])

            # store ID seperately(for log file name)
            self.playerID = str(self.playerInfo[0])

        else:
            # if 'cancel' button is clicked
            # close GUI window(this would result in fail of executing rest code)
            self.info_P = False

    def loadCard(self):

        # load image files names
        purple_card_files = glob.glob('imgs' + self.divider + 'purple_*.png')
        blue_card_files = glob.glob('imgs' + self.divider + 'blue*.png')
        black_card_files = glob.glob('imgs' + self.divider + 'black_*.png')
        self.target_card_files = glob.glob('imgs' + self.divider + 'T_*.png')

        # emerging all card file names into one dictionary data type
        self.color_dic = {'black': black_card_files,
                          'purple': purple_card_files,
                          'blue': blue_card_files}

        # generate random number(1~6) for 8 runs 
        self.NumSeqperRun = []

        for i in range(self.max_run):
            self.NumSeqperRun.append([random.randrange(1, 7) for _ in range(0, self.symbolsPerRun)])
        print self.NumSeqperRun

        # generate target card color sequence for each run
        self.targets = self.color * (self.max_run / len(self.color))
        random.shuffle(self.targets)

    def createSeq(self):

        # creat variable (list type) 
        self.non_targets = []
        self.seqPerRun = []

        for i in range(self.max_run):
            # copy data 
            # do not use 'non_target_color = self.color' (making changes to 'non_target_color' will also change 'self.color' )
            non_target_color = self.color[:]
            # get non-target colors
            non_target_color.remove(self.targets[i])

            # assign variables data
            self.non_targets.append(non_target_color)
            self.seqPerRun.append([self.targets[i]] * self.targetsPerRun +
                                  self.non_targets[i] * (self.non_targetsPerRun / 2))
            # randomize the element order inside the sequence
            random.shuffle(self.seqPerRun[i])

            #            print self.seqPerRun[i]
            #            print '---------------------'

    def createStim(self):

        tick = core.Clock()
        img_seq = []
        cardStimSeq_temp = []
        self.Card = []
        self.RunStim = []
        self.targetCard = []
        self.ask = []

        # make 2D list of Card image for all runs (number of run * number of symbols per run)
        for run in range(self.max_run):

            # get color&number sequence of current run 
            color_seq = self.seqPerRun[run]
            num_seq = self.NumSeqperRun[run]
            img_seq = []
            cardStimSeq_temp = []

            for symbol in range(self.symbolsPerRun):
                # get file name of 'symbol'th image in current sequence
                img_seq.append(self.color_dic.get(color_seq[symbol])[num_seq[symbol] - 1])
                # create stimuli type image according to its file name
                cardStimSeq_temp.append(
                    visual.ImageStim(self.win, image=img_seq[symbol], size=self.cardSize, pos=self.pos_symbol))

            # append stimuli type image sequence of current run to self.card
            self.Card.append(cardStimSeq_temp)

            # make stimuli for asking seen target screen
            txt = 'How many ' + self.targets[run] + ' card did you see?'
            self.ask.append(
                visual.TextStim(self.win, text=txt, wrapWidth=1.5, pos=self.pos_title, height=self.titleSize))

            # make stimuli for target screen
            # get image file name of target color card
            target = 'imgs' + self.divider + 'T_' + self.targets[run] + '.png'
            # get index of this file name stored in variable
            idx = self.target_card_files.index(target)
            # create stimuli according to this index
            self.targetCard.append(
                visual.ImageStim(self.win, image=self.target_card_files[idx], size=self.cardSize, pos=self.pos_symbol))
            # create text stimuli of 'Run X' (X:1~9)
            self.RunStim.append(
                visual.TextStim(self.win, text='Run ' + str(run + 1), pos=self.pos_title, height=self.titleSize))

        # make cross stimuli
        self.cross = visual.TextStim(self.win, text='+', pos=self.pos_symbol, height=self.titleSize)

        # make stimuli for asking seen target screen
        self.ans = visual.TextStim(self.win, text='', pos=self.pos_symbol, height=self.titleSize)
        self.error = visual.TextStim(self.win, text='', pos=self.pos_errorMsg, height=self.errorSize, color='orange')

        # make image stim for intro screen
        self.introduction = visual.ImageStim(self.win, image='intro.png', pos=self.pos_symbol, size=self.win_size,
                                             units='pix')

        print 'making stims took ', tick.getTime(), ' seconds'

    def initWindow(self):

        #        self.pp.setData(self.test_start)

        # something related to frame buffer, still trying to figure out what it is ...
        # visual.useFBO=True #if available (try without for comparison)

        # improve inter frame interval stability by changing the process priority
        if self.process_priority == 'normal':
            pass
        elif self.process_priority == 'high':
            core.rush(True)
        elif self.process_priority == 'realtime':
            # Only makes a diff compared to 'high' on Windows.
            core.rush(True, realtime=True)
        else:
            print 'Invalid process priority:', self.process_priority, "Process running at normal."
            self.process_priority = 'normal'

        self.win = visual.Window(fullscr=True, size=self.win_size, allowGUI=False, screen=0, monitor='testMonitor',
                                 color=self.bgcolor)
        self.win.setRecordFrameIntervals(True)

        # arise 'warning' level message if inter frame interval exceeds tolerance of 0.05ms
        self.win._refreshThreshold = 1 / 60.0 + 0.05
        logging.console.setLevel(logging.WARNING)

    def intro(self):
        self.clock = core.Clock()
        done = False

        while not done:

            self.introduction.draw()
            self.flipTimes.append(self.win.flip())

            input = event.getKeys(keyList=['return', 'space'])
            if input:
                done = True

    def runScreen(self):
        # this function is for showing target color card at the beginning of each run


        # create clock to monitor time starting from here
        self.clock.reset()
        self.addTimestampandTigger('run_start', self.run_start)
        self.win.callOnFlip(self.addTimestampandTigger, 'runScreen', self.runscreen)
        # display stimuli on screen for 'dur_target' secends
        while self.clock.getTime() < self.dur_target:
            self.targetCard[self.run - 1].draw()
            self.RunStim[self.run - 1].draw()
            self.flipTimes.append(self.win.flip())

    def crossScreen(self):
        # this function is for showing '+' symbol before card sequnce being shown

        self.win.callOnFlip(self.addTimestampandTigger, 'crossScreen', self.crossscreen)
        # reset clock time to 0 and start ticking from here
        self.clock.reset()
        while self.clock.getTime() < self.dur_cross:
            self.cross.draw()
            self.flipTimes.append(self.win.flip())

    def sequence(self):
        # this function is for showing sequence of card in each run

        # reset clock
        self.clock.reset()

        # loop flag
        done = False
        # pointer of sequnce index
        pointer = 0
        count = 0

        color_seq = self.seqPerRun[self.run - 1]

        #        self.pp.setData(self.run_start)



        while not done:
            # get current time
            t = self.clock.getTime()

            # when elements in the card sequence have not been all retrieved by pointer
            if pointer < self.symbolsPerRun - 1:

                if t <= self.step:
                    if t <= self.dur_symbol:
                        count += 1
                        self.Card[self.run - 1][pointer].draw()
                        if count == 1:
                            print color_seq[pointer]
                            if color_seq[pointer] == 'black':
                               if color_seq[pointer]==self.targets[self.run-1]:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'black_tg', self.black_t)
                               else:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'black_nt', self.black_nt)
                            elif color_seq[pointer] == 'purple':
                               if color_seq[pointer]==self.targets[self.run-1]:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'purple_tg',  self.purple_t)
                               else:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'purple_nt', self.purple_nt)
                            elif color_seq[pointer] == 'blue':
                               if color_seq[pointer]==self.targets[self.run-1]:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'blue_tg', self.blue_t)
                               else:
                                   self.win.callOnFlip(self.addTimestampandTigger, 'blue_nt', self.blue_nt)
                else:
                    # reset time
                    self.clock.reset()
                    count = 0
                    # point to next index of card sequence
                    pointer += 1
            # all elements have being retrieved
            else:
                # done loop
                done = True

            # update frame and store current updating time 
            self.flipTimes.append(self.win.flip())



            #        self.pp.setData(self.run_end)

    def userInp(self):
        # this function is for displaying a screen which subjects can input how many targets they seen in each run.

        self.win.callOnFlip(self.addTimestampandTigger, 'userInp_start',  self.userInp_start)

        answer = []

        # loop flag
        done = False

        # clear all pressed key in the buffer(only get pressed key after here)
        event.clearEvents()
        while not done:

            # get name of pressed key(only keys in the keylist will be recognized)
            inp = event.getKeys(keyList=self.keys)

            # subject pressed a number 
            if inp != [] and (inp[0] not in ['return', 'backspace']):
                # inp is list type (eg.'[u'1']'), inp[0] is string type(eg.'1')
                answer.append(inp[0])
                str = ''.join(answer)
                # update stimuli text
                self.ans.text = str

            # subject pressed backspace(to delete a number previously pressed)
            elif inp == ['backspace']:
                if answer != []:
                    # delete the number last pressed
                    del answer[-1]
                    str = ''.join(answer)

                self.ans.text = str

            # subject pressed enter(to finish input)
            elif inp == ['return']:
                # subject didn't input any number
                if answer == []:
                    self.error.text = 'Please input a number!'
                # finish loop
                else:
                    done = True

            self.ask[self.run - 1].draw()
            self.error.draw()
            self.ans.draw()

            self.flipTimes.append(self.win.flip())
        self.addTimestampandTigger('userInp_done', self.userInp_end)
        self.addTimestampandTigger('run_end', self.run_end)
        self.ans.text = ''
        self.error.text = ''
        self.userAnswer.append(str)

    #        self.pp.setData(self.userInp_end)

    def endScreen(self):
        # show 'finished' 
        self.win.callOnFlip(self.addTimestampandTigger, 'endScreen', self.endscreen)

        finish = visual.TextStim(self.win, text='Finished', pos=self.pos_symbol, height=self.titleSize)
        self.clock.reset()

        while self.clock.getTime() < 2.0:
            finish.draw()
            self.flipTimes.append(self.win.flip())
        self.win.close()

    #        self.pp.setData(self.test_end)

    def addLog(self):

        logwrite = open(self.Log, 'a')
        logwrite.write('subject seen targets: ' + str(self.userAnswer))

    def addTimestampandTigger(self, marker, trigger):

        timestamp_write = open(self.timestamp_File, 'a',)
        psychopytime = self.clock_ts.getTime()
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f ' + str(psychopytime) + ' ')

        timestamp_write.write(time + str(marker) + '\n')
        ctypes.windll.inpout32.Out32(self.pport_address, trigger)
        core.wait(0.005)
        ctypes.windll.inpout32.Out32(self.pport_address, 0)
        
    def plotframe(self):

        # calculate some values
        intervalsMS = pylab.array(self.win.frameIntervals) * 1000
        m = pylab.mean(intervalsMS)
        sd = pylab.std(intervalsMS)
        # se=sd/pylab.sqrt(len(intervalsMS)) # for CI of the mean
        nTotal = len(intervalsMS)
        nDropped = sum(intervalsMS > (1.5 * m))

        self.flipTimes = numpy.array(self.flipTimes)
        ifis = (self.flipTimes[1:] - self.flipTimes[:-1]) * 1000

        # plot the frameintervals
        pylab.figure(figsize=[12, 8], )

        pylab.subplot2grid((2, 2), (0, 0), colspan=2)
        pylab.plot(intervalsMS, '-')
        pylab.ylabel('t (ms)')
        pylab.xlabel('frame N')
        pylab.title("Dropped/Frames = %i/%i = %.3f%%. Process Priority: %s" % (
            nDropped, nTotal, 100 * nDropped / float(nTotal), self.process_priority), fontsize=12)
        #
        pylab.subplot2grid((2, 2), (1, 0))
        pylab.hist(intervalsMS, 50, normed=0, histtype='stepfilled')
        pylab.xlabel('t (ms)')
        pylab.ylabel('n frames')
        pylab.title("win.frameIntervals\nMean=%.2fms, s.d.=%.2f, 99%%CI(frame)=%.2f-%.2f" % (
            m, sd, m - 2.58 * sd, m + 2.58 * sd), fontsize=12)
        #
        pylab.subplot2grid((2, 2), (1, 1))
        pylab.hist(ifis, 50, normed=0, histtype='stepfilled')
        pylab.xlabel('t (ms)')
        pylab.ylabel('n frames')
        pylab.title("Inter Flip Intervals\nMean=%.2fms, s.d.=%.2f, range=%.2f-%.2f ms" % (
            ifis.mean(), ifis.std(), ifis.min(), ifis.max()), fontsize=12)

        pylab.tight_layout()
        pylab.savefig('framefig' + self.divider + self.filname + 'pretest_framefig')
        pylab.show()


if __name__ == '__main__':
    app = color_pretest()
    app.main()
# app.run()
