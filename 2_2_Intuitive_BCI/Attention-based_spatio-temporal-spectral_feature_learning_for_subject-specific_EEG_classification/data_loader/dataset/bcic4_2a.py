import numpy as np
import torch
from torch.utils.data import Dataset
from mne.filter import resample
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import Preprocessor as BraindPreprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.windowers import create_windows_from_events

from utils.utils import print_off, print_on
from data_loader.preprocessor import Preprocessor


class BCIC4_2A(Dataset):
    def __init__(self, options, phase):
        self.options = options
        self.load_data(phase)
        self.X, self.y = Preprocessor.label_selection(self.X, self.y, options.labels)
        self.segmentation()
        self.torch_form()

    def load_data(self, phase):
        X_bundle, y_bundle = [], []
        n_session = {'train': 1, 'test': 2}[phase]
        for band in self.options.band:
            X, y = get_bcic4_2a(self.options.subject, band[0], band[1],
                                start_time=self.options.start_time,
                                end_time=self.options.end_time,
                                split_session=True,
                                n_session=n_session,
                                verbose=self.options.verbose)
            X_bundle.append(X)
            y_bundle.append(y)
        self.X = np.stack(X_bundle, axis=1)
        self.y = np.array(y_bundle[0])

    def segmentation(self):
        self.X = Preprocessor.segment_tensor(self.X, self.options.window_size, self.options.step)

    def torch_form(self):
        self.X = torch.Tensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample


def get_bcic4_2a(subject,
                 low_hz=None,
                 high_hz=None,
                 start_time=-0.5,
                 end_time=None,
                 resampling=None,
                 split_session=True,
                 n_session=1,
                 verbose=False,
                 *args,
                 **kwargs):
    X = []
    y = []

    if isinstance(subject, int):
        subject = [subject]

    for subject_id in subject:
        # Load data
        if not verbose:
            print_off()
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

        # Preprocess
        factor_new = 1e-3
        init_block_size = 1000

        preprocessors = [
            # keep only EEG sensors
            BraindPreprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
            # convert from volt to microvolt
            BraindPreprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
            # bandpass filter
            BraindPreprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
            # exponential moving standardization
            BraindPreprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                               init_block_size=init_block_size, apply_on_array=True)
        ]
        preprocess(dataset, preprocessors)

        # Divide data by trial: Check sampling frequency
        sfreq = dataset.datasets[0].raw.info['sfreq']
        if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
            raise ValueError("Not match sampling rate.")

        trial_start_offset_samples = int(start_time * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True
        )

        # If you need split data, try this.
        if split_session:
            if n_session == 1:
                windows_dataset = windows_dataset.split('session')['session_T']
            else:
                windows_dataset = windows_dataset.split('session')['session_E']

        # Merge subject
        for trial in windows_dataset:
            X.append(trial[0])
            y.append(trial[1])

    X = np.array(X)
    y = np.array(y)

    if end_time is not None:
        len_time = end_time - start_time
        X = X[..., : int(len_time * sfreq)]

    if resampling is not None:
        X = resample(np.array(X, dtype=np.float64), resampling / sfreq)

    if not verbose:
        print_on()

    return X, y
