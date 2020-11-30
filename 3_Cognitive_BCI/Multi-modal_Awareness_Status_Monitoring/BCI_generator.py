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