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