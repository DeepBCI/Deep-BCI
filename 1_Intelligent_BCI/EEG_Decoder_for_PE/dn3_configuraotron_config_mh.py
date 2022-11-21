import mne
import yaml
from yamlinclude import YamlIncludeConstructor
import tqdm
import warnings
import mne.io as loader
import numpy as np
import scipy.signal as signal
from parse import search
from fnmatch import fnmatch
from pathlib import Path
from collections import OrderedDict
from mne import pick_types, read_annotations, set_log_level

from dn3_data_dataset_mh import Dataset, RawTorchRecording, EpochTorchRecording, Thinker, DatasetInfo, DumpedDataset
from dn3_utils_mh import make_epochs_from_raw, DN3ConfigException, skip_inds_from_bad_spans
from transforms_instance_mh import MappingDeep1010, TemporalInterpolation
from transforms_channels_mh import stringify_channel_mapping
# from dn3_configuratron_extensions_mh import MoabbDataset

# from moabb.datasets.download import get_dataset_path

## For merging event.
def merging_events(eegfile, new_id, timing = [1, 2], id = 6, replace_events = False): # a1, a2 -> 6...

    dd = mne.merge_events(eegfile, timing, id, replace_events=replace_events)
    where = []
    # new_id = [6, 7, 6, 6]
    for i in dd:
        if i[2] == 6:
            where.append(i[0])

    chn = [0 for _ in range(len(where))]  # np.zeros(len(where))
    new_for_merge = []
    for pair in zip(where, chn, new_id):
        # print(pair)
        new_for_merge.append(pair)

    # epoch_data = mne.Epochs(eegfile, new_for_merge)

    return new_for_merge # event

def _fif_raw_or_epoch(fname, preload=True):
    # See https://mne.tools/stable/generated/mne.read_epochs.html
    # TODO : epoch with eeglab 추가
    if str(fname).endswith('-epo.fif'):
        return mne.read_epochs(fname, preload=preload)
    # elif str(fname).endswith('-raw.set'):
    #     return loader.read_raw_eeglab(fname, preload=preload)
    else:
        return loader.read_raw_fif(fname, preload=preload)


_SUPPORTED_EXTENSIONS = {
    '.edf': loader.read_raw_edf,
    # FIXME: need to handle part fif files
    '.fif': _fif_raw_or_epoch,
    '.mff': loader.read_raw_egi, # TODO 221102

    # TODO: add much more support, at least all of MNE-python
    '.bdf': loader.read_raw_bdf,
    '.gdf': loader.read_raw_gdf,
    '.set': loader.read_raw_eeglab,
    '.mat': loader.read_raw_fieldtrip
}

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

# Since we are doing a lot of loading in the configuratron, this is nice to suppress some tedious information.
# Keep in mind, removing this might help debug data loading problems, `mne.set_log_level(True)` to counteract.
set_log_level(False)


class _DumbNamespace:
    def __init__(self, d: dict):
        self._d = d.copy()
        for k in d:
            if isinstance(d[k], dict):
                d[k] = _DumbNamespace(d[k])
            if isinstance(d[k], list):
                d[k] = [_DumbNamespace(d[k][i]) if isinstance(d[k][i], dict) else d[k][i] for i in range(len(d[k]))]
        self.__dict__.update(d)

    def keys(self):
        return list(self.__dict__.keys())

    def __getitem__(self, item):
        return self.__dict__[item]

    def as_dict(self):
        return self._d


def _adopt_auxiliaries(obj, remaining): # 보조들 채택...
    def namespaceify(v):
        if isinstance(v, dict):
            return _DumbNamespace(v)
        elif isinstance(v, list):
            return [namespaceify(v[i]) for i in range(len(v))]
        else:
            return v

    obj.__dict__.update({k: namespaceify(v) for k, v in remaining.items()})


class ExperimentConfig:
    """
    Parses DN3 configuration files. Checking the DN3 token for listed datasets.
    """
    def __init__(self, config_filename: str, adopt_auxiliaries=True):
        """
        Parses DN3 configuration files. Checking the DN3 token for listed datasets.

        Parameters
        ----------
        config_filename : str
                          String for path to yaml formatted configuration file
        adopt_auxiliaries : bool
                             For any additional tokens aside from DN3 and specified datasets, integrate them into this
                             object for later use. Defaults to True. This will propagate for the detected datasets.
        """
        with open(config_filename, 'r', encoding="UTF-8") as fio: # TODO 1028
            self._original_config = yaml.load(fio, Loader=yaml.FullLoader)
        working_config = self._original_config.copy()

        if 'Configuratron' not in working_config.keys():
            raise DN3ConfigException("Toplevel `Configuratron` not found in: {}".format(config_filename))
        if 'datasets' not in working_config.keys():
            raise DN3ConfigException("`datasets` not found in {}".format([k.lower() for k in
                                                                          working_config.keys()]))

        self.experiment = working_config.pop('Configuratron')

        ds_entries = working_config.pop('datasets')
        ds_entries = dict(zip(range(len(ds_entries)), ds_entries)) if isinstance(ds_entries, list) else ds_entries
        usable_datasets = list(ds_entries.keys())

        if self.experiment is None:
            self._make_deep1010 = dict()
            self.global_samples = None
            self.global_sfreq = None
            return_trial_ids = False
            preload = False
            relative_directory = None
        else:
            # If not None, will be used
            self._make_deep1010 = self.experiment.get('deep1010', dict())
            if isinstance(self._make_deep1010, bool):
                self._make_deep1010 = dict() if self._make_deep1010 else None
            self.global_samples = self.experiment.get('samples', None)
            self.global_sfreq = self.experiment.get('sfreq', None)
            usable_datasets = self.experiment.get('use_only', usable_datasets)
            preload = self.experiment.get('preload', False)
            return_trial_ids = self.experiment.get('trial_ids', False)
            relative_directory = self.experiment.get('relative_directory', None)

        self.datasets = dict()
        for i, name in enumerate(usable_datasets):
            if name in ds_entries.keys():
                self.datasets[name] = DatasetConfig(name, ds_entries[name], deep1010=self._make_deep1010,
                                                    samples=self.global_samples, sfreq=self.global_sfreq,
                                                    preload=preload, return_trial_ids=return_trial_ids,
                                                    relative_directory=relative_directory)
            else:
                raise DN3ConfigException("Could not find {} in datasets".format(name))

        print("Configuratron found {} datasets.".format(len(self.datasets), "s" if len(self.datasets) > 0 else ""))

        if adopt_auxiliaries:
            _adopt_auxiliaries(self, working_config)


class DatasetConfig:
    """
    Parses dataset entries in DN3 config
    """
    def __init__(self, name: str, config: dict, adopt_auxiliaries=True, ext_handlers=None, deep1010=None,
                 samples=None, sfreq=None, preload=False, return_trial_ids=False, relative_directory=None):
        """
        Parses dataset entries in DN3 config

        Parameters
        ----------
        name : str
               The name of the dataset specified in the config. Will be replaced if the optional `name` field is present
               in the config.
        config : dict
                The configuration entry for the dataset
        ext_handlers : dict, optional
                       If specified, should be a dictionary that maps file extensions (with dot e.g. `.edf`) to a
                       callable that returns a `raw` instance given a string formatted path to a file.
        adopt_auxiliaries : bool
                            Adopt additional configuration entries as object variables.
        deep1010 : None, dict
                   If `None` (default) will not use the Deep1010 to map channels. If a dict, will add this transform
                   to each recording, with keyword arguments from the dict.
        samples: int, None
                 Experiment level sample length, superceded by dataset-specific configuration
        sfreq: float, None
               Experiment level sampling frequency to be adhered to, this will be enforced if not None, ignoring
               decimate.
        preload: bool
                 Whether to preload recordings when creating datasets from the configuration. Can also be specified with
                 `preload` configuratron entry.
        return_trial_ids: bool
                 Whether to construct recordings that return trial ids.
        relative_directory: Path
                 Path to reference *toplevel* configuration entry to (if not an absolute path)

        """
        self._original_config = dict(config).copy()

        # Optional args set, these help define which are required, so they come first
        def get_pop(key, default=None):
            config.setdefault(key, default)
            return config.pop(key)

        # Epoching relevant options
        # self.tlen = get_pop('tlen')
        self.filename_format = get_pop('filename_format')
        if self.filename_format is not None and not fnmatch(self.filename_format, '*{subject*}*'):
            raise DN3ConfigException("Name format must at least include {subject}!")
        self.annotation_format = get_pop('annotation_format')
        self.tmin = get_pop('tmin')
        self._create_raw_recordings = self.tmin is None
        self.picks = get_pop('picks')
        if self.picks is not None and not isinstance(self.picks, list):
            raise DN3ConfigException("Specifying picks must be done as a list. Not {}.".format(self.picks))
        self.decimate = get_pop('decimate', 1)
        self.baseline = get_pop('baseline')
        if self.baseline is not None:
            self.baseline = tuple(self.baseline)
        self.bandpass = get_pop('bandpass')
        self.drop_bad = get_pop('drop_bad', False)
        self.events = get_pop('events')
        if self.events is not None:
            if not isinstance(self.events, (dict, list)):
                self.events = {0: self.events}
            elif isinstance(self.events, list):
                self.events = dict(zip(self.events, range(len(self.events))))
            self.events = OrderedDict(self.events)
        self.force_label = get_pop('force_label', False)
        self.chunk_duration = get_pop('chunk_duration')
        self.rename_channels = get_pop('rename_channels', dict())
        if not isinstance(self.rename_channels, dict):
            raise DN3ConfigException("Renamed channels must map new values to old values.")
        self.exclude_channels = get_pop('exclude_channels', list())
        if not isinstance(self.exclude_channels, list):
            raise DN3ConfigException("Excluded channels must be in a list.")

        # other options
        self.data_max = get_pop('data_max')
        self.data_min = get_pop('data_min')
        self.name = get_pop('name', name)
        self.dataset_id = get_pop('dataset_id')
        self.preload = get_pop('preload', preload)
        self.dumped = get_pop('pre-dumped')
        self.hpf = get_pop('hpf', None)
        self.lpf = get_pop('lpf', None)
        self.filter_data = self.hpf is not None or self.lpf is not None
        if self.filter_data:
            self.preload = True
        self.stride = get_pop('stride', 1)
        self.extensions = get_pop('file_extensions', list(_SUPPORTED_EXTENSIONS.keys()))
        self.exclude_people = get_pop('exclude_people', list())
        self.exclude_sessions = get_pop('exclude_sessions', list())
        self.exclude = get_pop('exclude', dict())
        self.deep1010 = deep1010
        if self.deep1010 is not None and (self.data_min is None or self.data_max is None):
            print("Warning: Can't add scale index with dataset that is missing info.")
        self._different_deep1010s = list()
        self._targets = get_pop('targets', None)
        self._unique_events = set()
        self.return_trial_ids = return_trial_ids
        self.from_moabb = get_pop('moabb')

        self._samples = get_pop('samples', samples)
        self._sfreq = sfreq
        if sfreq is not None and self.decimate > 1:
            print("{}: No need to specify decimate ({}) when sfreq is set ({})".format(self.name, self.decimate, sfreq))
            self.decimate = 1

        # Funky stuff
        self._on_the_fly = get_pop('load_onthefly', False)

        # Required args
        # TODO refactor a bit
        try:
            self.toplevel = get_pop('toplevel')
            if self.toplevel is None:
                if self.from_moabb is None:
                    raise KeyError()
                else:
                    # TODO resolve the use of MOABB `get_dataset_path()` confusion with "signs" vs. name of dataset
                    self.toplevel = mne.get_config('MNE_DATA', default='~/mne_data')
            self.toplevel = self._determine_path(self.toplevel, relative_directory)
            self.toplevel = Path(self.toplevel).expanduser()
            self.tlen = config.pop('tlen') if self._samples is None else None
        except KeyError as e:
            raise DN3ConfigException("Could not find required value: {}".format(e.args[0]))
        if not self.toplevel.exists():
            raise DN3ConfigException("The toplevel {} for dataset {} does not exists".format(self.toplevel, self.name))

        # The rest
        if adopt_auxiliaries and len(config) > 0:
            print("Adding additional configuration entries: {}".format(config.keys()))
            _adopt_auxiliaries(self, config)

        self._extension_handlers = _SUPPORTED_EXTENSIONS.copy()
        if ext_handlers is not None:
            for ext in ext_handlers:
                self.add_extension_handler(ext, ext_handlers[ext])

        self._excluded_people = list()

        # Callbacks and custom loaders
        self._custom_thinker_loader = None
        self._thinker_callback = None
        self._custom_raw_loader = None
        self._session_callback = None

        # Extensions
        if self.from_moabb is not None:
            print("MOABB configuration : Not implemented")
            # try:
            #     self.from_moabb = MoabbDataset(self.from_moabb.pop('name'), self.toplevel.resolve(), **self.from_moabb)
            # except KeyError:
            #     raise DN3ConfigException("MOABB configuration is incorrect. Make sure to use 'name' under MOABB to "
            #                              "specify a compatible dataset.")
            # self._custom_raw_loader = self.from_moabb.get_raw

    _PICK_TYPES = ['meg', 'eeg', 'stim', 'eog', 'ecg', 'emg', 'ref_meg', 'misc', 'resp', 'chpi', 'exci', 'ias', 'syst',
                   'seeg', 'dipole', 'gof', 'bio', 'ecog', 'fnirs', 'csd', ]

    @staticmethod
    def _picks_as_types(picks):
        if picks is None:
            return False
        for pick in picks:
            if not isinstance(pick, str) or pick.lower() not in DatasetConfig._PICK_TYPES:
                return False
        return True

    @staticmethod
    def _determine_path(toplevel, relative_directory=None):
        if relative_directory is None or str(toplevel)[0] in '/~.':
            return toplevel
        return str((Path(relative_directory) / toplevel).expanduser())

    def add_extension_handler(self, extension: str, handler):
        """
        Provide callable code to create a raw instance from sessions with certain file extensions. This is useful for
        handling of custom file formats, while preserving a consistent experiment framework.

        Parameters
        ----------
        extension : str
                   An extension that includes the '.', e.g. '.csv'
        handler : callable
                  Callback with signature f(path_to_file: str) -> mne.io.Raw, mne.io.Epochs

        """
        assert callable(handler)
        self._extension_handlers[extension] = handler

    def scan_toplevel(self):
        """
        Scan the provided toplevel for all files that may belong to the dataset.

        Returns
        -------
        files: list
               A listing of all the candidate filepaths (before excluding those that match exclusion criteria).
        """
        files = list()
        pbar = tqdm.tqdm(self.extensions,
                         desc="Scanning {}. If there are a lot of files, this may take a while...".format(
                             self.toplevel))
        files = self.toplevel.glob("**/*{}".format(self.extensions))
        # for extension in pbar:
        #     pbar.set_postfix(dict(extension=extension))
        #     # files += self.toplevel.glob("**/*{}".format(extension))
        #     files += self.toplevel.glob("**/*{}".format(extension))
        return files

    def _is_narrowly_excluded(self, person_name, session_name):
        if person_name in self.exclude.keys():
            if self.exclude[person_name] is None:
                self._excluded_people.append(person_name)
                return True
            assert isinstance(self.exclude[person_name], dict)
            if session_name in self.exclude[person_name].keys() and self.exclude[person_name][session_name] is None:
                return True
        return False

    def is_excluded(self, f: Path, person_name, session_name):
        if self._is_narrowly_excluded(person_name, session_name):
            return True

        if True in [fnmatch(person_name, pattern) for pattern in self.exclude_people]:
            self._excluded_people.append(person_name)
            return True

        session_exclusion_patterns = self.exclude_sessions.copy()
        if self.annotation_format is not None:
            # Some hacks over here, but it will do
            patt = self.annotation_format.format(subject='**', session='*')
            patt = patt.replace('**', '*')
            patt = patt.replace('**', '*')
            session_exclusion_patterns.append(patt)
        for exclusion_pattern in session_exclusion_patterns:
            for version in (f.stem, f.name):
                if fnmatch(version, exclusion_pattern):
                    return True
        return False

    def _get_person_name(self, f: Path):
        if self.filename_format is None:
            person = f.parent.parent.name
        else:
            person = search(self.filename_format, str(f))
            if person is None:
                raise DN3ConfigException("Could not find person in {} using {}.".format(f.name, self.filename_format))
            person = person['subject']
        return person

    def _get_session_name(self, f: Path):
        if self.filename_format is not None and fnmatch(self.filename_format, "*{session*}*"):
            sess_name = search(self.filename_format, str(f))['session']
        else:
            sess_name = f.parent.name
        return sess_name

    def auto_mapping(self, files=None, reset_exclusions=True):
        """
        Generates a mapping of sessions and people of the dataset, assuming files are stored in the structure:
        `toplevel`/(*optional - <version>)/<person-id>/<session-id>.{ext}

        Parameters
        -------
        files : list
                Optional list of files (convertible to `Path` objects, e.g. relative or absolute strings) to be used.
                If not provided, will use `scan_toplevel()`.

        Returns
        -------
        mapping : dict
                  The keys are of all the people in the dataset, and each value another similar mapping to that person's
                  sessions.
        """
        if reset_exclusions:
            self._excluded_people = list()

        files = self.scan_toplevel() if files is None else files
        mapping = dict()
        for sess_file in files:
            sess_file = Path(sess_file)
            try:
                person_name = self._get_person_name(sess_file)
                session_name = self._get_session_name(sess_file)
            except DN3ConfigException:
                continue

            if self.is_excluded(sess_file, person_name, session_name):
                continue

            if person_name in mapping:
                mapping[person_name].append(str(sess_file))
            else:
                mapping[person_name] = [str(sess_file)]

        return mapping
    # [raw.ch_names[i] for i in picks], xform.mapping.numpy(), [raw.ch_names[i] for i in range(len(raw.ch_names)) if i not in picks]
    def _add_deep1010(self, ch_names: list, deep1010map: np.ndarray, unused):
        for i, (old_names, old_map, unused, count) in enumerate(self._different_deep1010s):
            if np.all(deep1010map == old_map):
                self._different_deep1010s[i] = (old_names, old_map, unused, count+1)
                return
        self._different_deep1010s.append((ch_names, deep1010map, unused, 1))

    def add_custom_raw_loader(self, custom_loader):
        """
        This is used to provide a custom implementation of taking a filename, and returning a :any:`mne.io.Raw()`
        instance. If properly constructed, all further configuratron options, such as resampling, epoching, filtering
        etc. should occur automatically.

        This is used to load unconventional files, e.g. '.mat' files from matlab, or custom '.npy' arrays, etc.

        Notes
        -----
        Consider using :any:`mne.io.Raw.add_events()` to integrate otherwise difficult (for the configuratron) to better
        specify events for each recording.

        Parameters
        ----------
        custom_loader: callable
                       A function that expects a single :any:`pathlib.Path()` instance as argument and returns an
                       instance of :any:`mne.io.Raw()`. To gracefully ignore problematic sessions, raise
                       :any:`DN3ConfigException` within.

        """
        assert callable(custom_loader)
        self._custom_raw_loader = custom_loader

    def add_progress_callbacks(self, session_callback=None, thinker_callback=None):
        """
        Add callbacks to be invoked on successful loading of session and/or thinker. Optionally, these can modify the
        respective loaded instances.

        Parameters
        ----------
        session_callback:
                          A function that expects a single session argument and can modify the (or return an
                          alternative) session.

        thinker_callback:
                          The same as for session, but with Thinker instances.

        """
        self._session_callback = session_callback
        self._thinker_callback = thinker_callback

    def _load_raw(self, path: Path):
        if self._custom_raw_loader is not None:
            return self._custom_raw_loader(path)
        if path.suffix in self._extension_handlers:
            # if path.suffix == '.mat':
            #     info = DatasetInfo(self.name, self.data_max, self.data_min, self._excluded_people,
            #                        targets=self._targets if self._targets is not None else len(self._unique_events))
            #     return self._extension_handlers[path.suffix](str(path), info)
            # else:
            self.preload = False #False #True # False # TODO 221109 hj
            if self.name == 'SPE' or self.name == 'RPE' or self.name == 'mbmf' or self.name == 'behavior' or self.name == 'pmb': # TODO 221114
                self.preload = True
            return self._extension_handlers[path.suffix](str(path), preload=self.preload)
        print("Handler for file {} with extension {} not found.".format(str(path), path.suffix))
        for ext in path.suffixes:
            if ext in self._extension_handlers:
                print("Trying {} instead...".format(ext))
                return self._extension_handlers[ext](str(path), preload=self.preload)

        raise DN3ConfigException("No supported/provided loader found for {}".format(str(path)))

    @staticmethod
    def _prepare_session(raw, tlen, decimate, desired_sfreq, desired_samples, picks, exclude_channels, rename_channels,
                         hpf, lpf): # TODO HERE 오래걸림
        # if hpf is not None or lpf is not None: # TODO 221109
        #     raw = raw.filter(hpf, lpf)
        # raw = raw.notch_filter(60, notch_widths=10) # TODO 221006 notch filter

        lowpass = raw.info.get('lowpass', None) # 220928 None
        raw_sfreq = raw.info['sfreq']
        new_sfreq = raw_sfreq / decimate if desired_sfreq is None else desired_sfreq

        # Don't allow violation of Nyquist criterion if sfreq is being changed # TODO : 221109
        # if lowpass is not None and (new_sfreq < 2 * lowpass) and new_sfreq != raw_sfreq:
        #     raise DN3ConfigException("Could not create raw for {}. With lowpass filter {}, sampling frequency {} and "
        #                              "new sfreq {}. This is going to have bad aliasing!".format(raw.filenames[0],
        #                                                                                         raw.info['lowpass'],
        #                                                                                         raw.info['sfreq'],
        #                                                                                         new_sfreq))

        # Leverage decimation first to match desired sfreq (saves memory)
        if desired_sfreq is not None:
            while (raw_sfreq // (decimate + 1)) >= new_sfreq:
                decimate += 1

        # Pick types
        picks = pick_types(raw.info, **{t: t in picks for t in DatasetConfig._PICK_TYPES}) \
            if DatasetConfig._picks_as_types(picks) else list(range(len(raw.ch_names)))

        # Exclude channel index by pattern match
        picks = ([idx for idx in picks if True not in [fnmatch(raw.ch_names[idx], pattern)
                                                       for pattern in exclude_channels]])

        # Rename channels
        renaming_map = dict()
        for new_ch, pattern in rename_channels.items(): #no
            for old_ch in [raw.ch_names[idx] for idx in picks if fnmatch(raw.ch_names[idx], pattern)]:
                renaming_map[old_ch] = new_ch
        try: # TODO raw.filenames 확인 -> raw단위 확인.
            raw = raw.rename_channels(renaming_map)
        except ValueError as e:
            print("Error renaming channels from session: ", raw.filenames[0])
            print("Failed to rename ", raw.ch_names, " using ", renaming_map)
            raise DN3ConfigException("Skipping channel name issue.")

        tlen = desired_samples / new_sfreq if tlen is None else tlen

        # TODO 221031 -> 221109
        # try:
        #     ch_n = raw._data.shape[0] # 21, 91000
        # except:
        #     ch_n = raw.load_data()._data.shape[0]
        #
        # original_timepoint = raw._data.shape[1] # 91000
        # f, t, Sxx = signal.spectrogram(raw._data, nperseg=256, window='hamming', fs=256) # 0번째.. -> load
        # # t: (406, ), f: (129, ), Sxx : (21, 129, 406) : (channel, freq, time) nperseg = 256
        # # t: (202, ), f: (257, ), Sxx : (21, 257, 202) : (channel, freq, time) nperseg = 512
        # e = np.where(f >= 50)
        # freq = f[0:e[0][0]]
        # tfreq = Sxx[:, 0:e[0][0], :] # (21, 50, 406)
        # # raw._data = tfreq.reshape(ch_n, -1) # (21, 20300)
        # raw._data = tfreq.reshape(-1, len(t))  # (3200, 1769) = 64 x 50, 1769 # TODO 221109
        #
        # # import matplotlib.pyplot as plt
        # # plt.pcolormesh(t[750:1300], freq, tfreq[0, :, 750:1300], shading='gouraud')
        # # plt.ylabel('Frequency [Hz]')
        # # plt.xlabel('Time [sec]')
        # # plt.show()
        #
        # changed_timepoint = raw._data.shape[1] # 20300
        # # raw.n_times = raw.n_times * changed_tproimepoint / original_timepoint
        # changed_ratio = changed_timepoint / original_timepoint

        # return raw, tlen, picks, new_sfreq, changed_ratio, decimate # TODO 221109
        return raw, tlen, picks, new_sfreq, decimate  # TODO 221109
        # return raw, tlen, picks, new_sfreq

    def _construct_session_from_config(self, session, sess_id, thinker_id): # TODO HERE 오래걸림
        bad_spans = None
        if thinker_id in self.exclude.keys():
            if sess_id in self.exclude[thinker_id].keys():
                bad_spans = self.exclude[thinker_id][sess_id]
                if bad_spans is None:
                    raise DN3ConfigException("Skipping {} - {}".format(thinker_id, sess_id))

        def load_and_prepare(sess):
            if not isinstance(sess, Path):
                sess = Path(sess)
            r = self._load_raw(sess)
            # try:
            #     r = self._load_raw(sess)
            # except:
            #     print(sess) # subj0057
            return (sess, *self._prepare_session(r, self.tlen, self.decimate, self._sfreq, self._samples, self.picks,
                                                self.exclude_channels, self.rename_channels, self.hpf, self.lpf))
        # sess, raw, tlen, picks, new_sfreq, changed_ratio, decimate = load_and_prepare(session) # TODO 221031 221109
        sess, raw, tlen, picks, new_sfreq, decimate = load_and_prepare(session)  # TODO 221031 221109
        self.decimate = decimate # TODO 221109


        # Fixme - deprecate the decimate option in favour of specifying desired sfreq's
        if self._create_raw_recordings:
            if self._on_the_fly:
                recording = RawOnTheFlyRecording(raw, tlen, lambda s: load_and_prepare(s)[1], stride=self.stride,
                                                 decimate=self.decimate, ch_ind_picks=picks, bad_spans=bad_spans)
            else: # here for pretraining
                # recording = RawTorchRecording(raw, tlen, stride=self.stride, decimate=self.decimate, ch_ind_picks=picks,
                #                               bad_spans=bad_spans, time_2_freq = changed_ratio) # TODO 221031
                recording = RawTorchRecording(raw, tlen, stride=self.stride, decimate=self.decimate, ch_ind_picks=picks,
                                              bad_spans=bad_spans)  # TODO 221031
        else:
            use_annotations = self.events is not None and True in [isinstance(x, str) for x in self.events.keys()]
            if not isinstance(raw, (mne.Epochs, mne.epochs.EpochsFIF)):  # Annoying other epochs type
                if use_annotations and self.annotation_format is not None:
                    patt = self.annotation_format.format(subject=thinker_id, session=sess_id)
                    ann = [str(f) for f in session.parent.glob(patt)]
                    if len(ann) > 0:
                        if len(ann) > 1:
                            print("More than one annotation found for {}. Falling back to {}".format(patt, ann[0]))
                        raw.set_annotations(read_annotations(ann[0]))

                if isinstance(raw, mne.io.edf.edf.RawGDF) :
                    self.events = mne.events_from_annotations(raw)[1]
                if self.name == 'SPE' :
                    if self.hpf is not None or self.lpf is not None:  # TODO 221114
                        raw = raw.filter(self.hpf, self.lpf)
                    # self.events = merging_events(raw, new_id) # eegfile, new_id, timing = [1, 2], id = 6, replace_events = False
                    self.events = {'0': 0, '1': 1}
                elif self.name == 'RPE' :
                    if self.hpf is not None or self.lpf is not None:  # TODO 221114
                        raw = raw.filter(self.hpf, self.lpf)
                    # merging_events
                    self.events = {'0': 0, '1': 1}
                elif self.name == 'mbmf' :
                    if self.hpf is not None or self.lpf is not None:  # TODO 221114
                        raw = raw.filter(self.hpf, self.lpf)
                    # merging_events
                    self.events = {'0': 0, '1': 1} # 0 : mf, 1: mb
                elif self.name == 'behavior' : # TODO
                    if self.hpf is not None or self.lpf is not None:  # TODO 221114
                        raw = raw.filter(self.hpf, self.lpf)
                    # merging_events
                    self.events = {'0': 0, '1': 1} # 0 : mf, 1: mb
                elif self.name == 'pmb' : # TODO 221024
                    if self.hpf is not None or self.lpf is not None:  # TODO 221114
                        raw = raw.filter(self.hpf, self.lpf)
                    # merging_events
                    self.events = {'0': 0, '1': 1} # 0 : mf, 1: mb
                epochs = make_epochs_from_raw(raw, self.tmin, tlen, event_ids=self.events, baseline=self.baseline,
                                              decim=self.decimate, filter_bp=self.bandpass, drop_bad=self.drop_bad,
                                              use_annotations=use_annotations, chunk_duration=self.chunk_duration, name = self.name) #raw : (64, 396359) # epochs : (164, 64, 1167) = event, channel, time
            else:
                epochs = raw

            event_map = {v: v for v in self.events.values()} if use_annotations else self.events # epochs 확인 - 82 - 164

            # if isinstance(raw, mne.io.edf.edf.RawGDF) and event_map == None :
            #     event_map = mne.events_from_annotations(raw)[1]

            # bad spans 활용 -> 안할거임
            """subj006-sess1, subj012-sess2, subj018-sess1"""

            # for masking # TODO 221006 masking try.... ㅎ
            events_from_annotation_, event_dict_ = mne.events_from_annotations(raw) # epochs.events = where, chn, new_id
            where_a1 = []
            where_a2 = []
            for i in events_from_annotation_:
                if i[2] == event_dict_['act1']:
                    where_a1.append(i[0])
                elif i[2] == event_dict_['act2']:
                    where_a2.append(i[0])

            for i in range(len(epochs)):
                if epochs[i].events[0][0] in where_a1 : #== 해당 epoch = act1일 때,
                    index_a = where_a1.index(epochs[i].events[0][0])
                    when_a2 = where_a2[index_a]
                    if epochs[i].events[0][0] + 2500 > when_a2:
                        epochs._data[i][:, when_a2 - epochs[i].events[0][0] : ] = 0 # 64 by 3500 # act2 후는 0으로 마스킹



            self._unique_events = self._unique_events.union(set(np.unique(epochs.events[:, -1])))



            recording = EpochTorchRecording(epochs, ch_ind_picks=picks, event_mapping=event_map,
                                            force_label=self.force_label,
                                            skip_epochs=skip_inds_from_bad_spans(epochs, bad_spans))

        if len(recording) == 0:
            raise DN3ConfigException("The recording at {} has no viable training data with the configuration options "
                                     "provided. Consider excluding this file or changing parameters.".format(str(session
                                                                                                                 )))

        if self.deep1010 is not None:
            # FIXME dataset not fully formed, but we can hack together something for now
            _dum = _DumbNamespace(dict(channels=recording.channels, info=dict(data_max=self.data_max,
                                                                              data_min=self.data_min))) # recording.channels = 90
            xform = MappingDeep1010(_dum, **self.deep1010) # 60 by 90
            recording.add_transform(xform)
            self._add_deep1010([raw.ch_names[i] for i in picks], xform.mapping.numpy(),
                               [raw.ch_names[i] for i in range(len(raw.ch_names)) if i not in picks]) # ch_names: list, deep1010map: np.ndarray, unused

        if recording.sfreq != new_sfreq:
            new_sequence_len = int(tlen * new_sfreq) if self._samples is None else self._samples
            recording.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=new_sfreq))

        return recording

    def add_custom_thinker_loader(self, thinker_loader):
        """
        Add custom code to load a specific thinker from a set of session files.

        Warnings
        ----------
        For all intents and purposes, this circumvents most of the configuratron, and results in it being mostly
        a tool for organizing dataset files. Most of the options are not leveraged and must be implemented by the
        custom loader. Please open an issue if you'd like to develop this option further!

        Parameters
        ----------
        thinker_loader:
                        A function that takes a dict argument that consists of the session-ids that map to filenames
                        (str) of all the detected session for the given thinker and a second argument for the detected
                        name of the person. The function should return a single instance of type :any:`Thinker`.
                        To gracefully ignore the person, raise a :any:`DN3ConfigException`

        """
        self._custom_thinker_loader = thinker_loader

    def _construct_thinker_from_config(self, thinker: list, thinker_id): # TODO HERE 오래 걸림
        sessions = {self._get_session_name(Path(s)): s for s in thinker}
        if self._custom_thinker_loader is not None:
            thinker = self._custom_thinker_loader(sessions, thinker_id)
        else:
            sessions = dict() #here
            for sess in thinker:
                sess = Path(sess)
                sess_name = self._get_session_name(sess)
                try:
                    new_session = self._construct_session_from_config(sess, sess_name, thinker_id)
                    after_cb = None if self._session_callback is None else self._session_callback(new_session)
                    sessions[sess_name] = new_session if after_cb is None else after_cb
                except DN3ConfigException as e:
                    tqdm.tqdm.write("Skipping {}. Exception: {}.".format(sess_name, e.args))
            if len(sessions) == 0:
                raise DN3ConfigException
            thinker = Thinker(sessions)

        if self.deep1010 is not None:
            # Quick check for if Deep1010 was already added to sessions
            skip = False
            for s in thinker.sessions.values():
                if skip:
                    break
                for x in s._transforms:
                    if isinstance(x, MappingDeep1010):
                        skip = True
                        break
            if not skip: # no
                # FIXME dataset not fully formed, but we can hack together something for now
                og_channels = list(thinker.channels[:, 0])
                _dum = _DumbNamespace(dict(channels=thinker.channels, info=dict(data_max=self.data_max,
                                                                                            data_min=self.data_min)))
                xform = MappingDeep1010(_dum, **self.deep1010)
                thinker.add_transform(xform)
                self._add_deep1010(og_channels, xform.mapping.numpy(), [])

        if self._sfreq is not None and thinker.sfreq != self._sfreq: # no
            new_sequence_len = int(thinker.sequence_length * self._sfreq / thinker.sfreq) if self._samples is None \
                else self._samples
            thinker.add_transform(TemporalInterpolation(new_sequence_len, new_sfreq=self._sfreq))

        return thinker

    def auto_construct_dataset(self, mapping=None, **dsargs):
        """
        This creates a dataset using the config values. If tlen and tmin are specified in the config, creates epoched
        dataset, otherwise Raw.

        Parameters
        ----------
        mapping : dict, optional
                A dict specifying a list of sessions (as paths to files) for each person_id in the dataset. e.g.
                {
                  person_1: [sess_1.edf, ...],
                  person_2: [sess_1.edf],
                  ...
                }
                If not specified, will use `auto_mapping()` to generate.
        dsargs :
                Any additional arguments to feed for the creation of the dataset. i.e. keyword arguments to `Dataset`'s
                constructor (which id's to return). If `dataset_info` is provided here, it will override what was
                inferrable from the configuration file.

        Returns
        -------
        dataset : Dataset
                An instance of :any:`Dataset`, constructed according to mapping.
        """
        if self.dumped is not None:
            path = Path(self.dumped)
            if path.exists():
                tqdm.tqdm.write("Found pre-dumped dataset directory at {}".format(self.dumped))
                info = DatasetInfo(self.name, self.data_max, self.data_min, self._excluded_people,
                                   targets=self._targets if self._targets is not None else len(self._unique_events))
                dataset = DumpedDataset(path, info=info)
                tqdm.tqdm.write(str(dataset))
                return dataset
            else:
                tqdm.tqdm.write("Could not load pre-dumped data, falling back to original data...")

        if self.from_moabb:
            print("Creating dataset using MOABB...")
            print("NOT IMPLEMENTED MOABB")
            # mapping = self.from_moabb.get_pseudo_mapping(exclusion_cb=self.is_excluded)
            # print("Converting MOABB format to DN3")
        if mapping is None:
            return self.auto_construct_dataset(self.auto_mapping(), **dsargs)

        file_types = "Raw" if self._create_raw_recordings else "Epoched"
        if self.preload:
            file_types = "Preloaded " + file_types
        print("Creating dataset of {} {} recordings from {} people.".format(sum(len(mapping[p]) for p in mapping),
                                                                            file_types, len(mapping)))
        description = "Loading {}".format(self.name)
        thinkers = dict()
        for t in tqdm.tqdm(mapping, desc=description, unit='person'): # mapping : subj32 : path, ....
            try:
                new_thinker = self._construct_thinker_from_config(mapping[t], t) # thinker: list, thinker_id
                after_cb = None if self._thinker_callback is None else self._thinker_callback(new_thinker)
                thinkers[t] = new_thinker if after_cb is None else after_cb
            except DN3ConfigException:
                tqdm.tqdm.write("None of the sessions for {} were usable. Skipping...".format(t))

        info = DatasetInfo(self.name, self.data_max, self.data_min, self._excluded_people,
                           targets=self._targets if self._targets is not None else len(self._unique_events))
        dsargs.setdefault('dataset_info', info)
        dsargs.setdefault('dataset_id', self.dataset_id)
        dsargs.setdefault('return_trial_id', self.return_trial_ids)
        dataset = Dataset(thinkers, **dsargs) # 90 channels
        print(dataset) #thinker로 이루어진 dataset
        if self.deep1010 is not None: # return_mask : true
            print("Constructed {} channel maps".format(len(self._different_deep1010s)))
            for names, deep_mapping, unused, count in self._different_deep1010s: # 63 by 4 for lemon202
                print('=' * 20)
                print("Used by {} recordings:".format(count))
                try:
                    print(stringify_channel_mapping(names, deep_mapping)) #names : 59, deep_mapping : 59 by 90
                except:
                    print("?")
                print('-'*20)
                print("Excluded {}".format(unused))
                print('=' * 20)
        #     dataset.add_transform(MappingDeep1010(dataset))
        return dataset


class RawOnTheFlyRecording(RawTorchRecording):

    def __init__(self, raw, tlen, file_loader, session_id=0, person_id=0, stride=1, ch_ind_picks=None,
                 decimate=1, **kwargs):
        """
        This provides a workaround for the normal raw recording pipeline so that files are not loaded in any way until
        they are needed. MNE's Raw object are too bloated for extremely large datasets, even without preloading.

        Parameters
        ----------
        raw
        tlen
        file_loader
        session_id
        person_id
        stride
        ch_ind_picks
        decimate
        kwargs
        """
        super().__init__(raw, tlen, session_id, person_id, stride, ch_ind_picks, decimate, **kwargs)
        self.file_loader = file_loader

    def _raw_workaround(self, raw):
        return

    @property
    def raw(self):
        with warnings.catch_warnings():
            # Ignore loading warnings that were already addressed during configuratron start-up
            warnings.simplefilter("ignore")
            return self.file_loader(self.filename)

    def preprocess(self, preprocessor, apply_transform=True):
        result = preprocessor(recording=self)
        if result is not None:
            raise DN3ConfigException("Modifying raw after preprocessing is not supported with on-the-fly")
        if apply_transform:
            self.add_transform(preprocessor.get_transform())
