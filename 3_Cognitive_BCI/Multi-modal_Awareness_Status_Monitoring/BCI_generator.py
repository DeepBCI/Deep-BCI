class BCIDataGenerator:
    def __init__(self, filename, window=4, shift=2, cv_fold=5, preload=False):
        self.cv_fold = cv_fold

        self.data_source = h5py.File(filename,'r')
        self.data = []      # empty list to hold signal data of all 19 subjects in order.
        self.num_class = 3  # awake, drowsy, sleep

        x_merge = []
        y_merge = []
        c_merge = []
        s_merge = []

        ppg_window = 250 * window   # PPG signal is sampled by 250Hz. if window is 4sec, ppg_window is 1000point window.
        ppg_shift = 250 * shift

        # slice shape(1, (k-1)*ceil(m/k)) array to index m, and return the array after randomly rearranging.
        cross_validation_index = lambda m, k: np.random.permutation(np.tile(np.arange(k), np.math.ceil(m / k))[:m])

        subject_info = self.data_source['subject_info']
        for six in range(subject_info['name'].shape[0]):  # repeat this for all subject.
            if preload:  # default value of preload=False
                # append each subject's dataset object to data list.
                # f'{1:02}' is simply to make string '01', which is one of keys of group.
                self.data.append(self.data_source['data'][f'{six:02}']['ppg'])
            stimulus_info = self.data_source['stimulus_info'][f'{six:02}']
            x = []
            for tix in range(stimulus_info['ppg'].shape[0]):
                p1 = 0 if tix is 0 else stimulus_info['ppg'][tix - 1]
                p2 = stimulus_info['ppg'][tix]
                # range(p1, p2-ppg_window,ppg_shift) is a list that contains points of ppg from p1 to p2. step is shift(default=2)sec.
                # map change elements in range to tuple of (dix, next dix). list change tuple to list. np.stack put them in one list
                x.append(
                    np.stack(list(map(lambda dix: (dix, dix + ppg_window), range(p1, p2 - ppg_window, ppg_shift)))))
                # return string that shows whether sbj responsed(0) or not(1).=>'00000110111...'
            resp_str = ''.join(map(str, np.isinf(stimulus_info['resp']).astype(int)))
            lab = np.full((len(x)), np.nan)  # return array of length len(x), filled with nan
            # finditer finds index of string where 0 is repeated same or more than 3
            for loc in re.finditer('0{3,}', resp_str):
                lab[loc.start() + 1:loc.end()] = 0  # this lis labeled as 0, because it means awake.
                # if 6 0 in row, change 5 of them except first one
            for loc in re.finditer('1{3,}', resp_str):
                lab[loc.start() + 1:loc.end()] = 1  # this is labeled as 1, because it means sleeping
                # if 6 1 in row, change 5 of them except first one
            for loc in re.finditer('000111', resp_str):
                lab[loc.start() + 2:loc.end() - 1] = 2  # this is labeled as 2, because it means drowsy
                # if 000111, it is changed to 002221