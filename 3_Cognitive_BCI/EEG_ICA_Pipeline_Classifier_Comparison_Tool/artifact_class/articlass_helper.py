import pickle
import os
import json
import hdf5storage as h5s
import numpy as np

"""Collection of helper functions to run the ica comparison training pipeline"""

def load_matfile(datafile):

    datamat = h5s.loadmat(datafile)
    return datamat


def setup_directory(destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    return


def load_pickle(pkfile):
    with open(pkfile, 'rb') as f:
        re_pkl = pickle.load(f)
    return re_pkl


def remove_eog(x, eog_pos, till_end=False):
    """
    Assumes x is: time x eeg x sample
    Support selective erasing of channel indices, doesn't really have to be eog but
    in most cases we only use them for eogs.
    Take care that there's a hard assumption that eeg channel dimension is always at the second dimension.
    :param x:
    :param eog_pos:
    :return:
    """
    ##x = x[:, 0:eog_pos, :]
    if till_end is True:
        x = x[:, 0:eog_pos, :]
    else:
        x = np.delete(x, eog_pos, 1)
    return x

def trim_trials(x, y, trim_idx):
    '''
    assumes x is preprocessed, and that trim idx shows idx of samples to save
    trim idx is defined in the main script
    :param x:
    :param y:
    :param trim_idx:
    :return:
    '''

    new_x = x[trim_idx]
    new_y = y[:, trim_idx]

    return new_x, new_y



def preproc_x(x, remove_eog_opt=False, eog_positions=59, to_4d=False):
    """
    x dimension at input is time x eeg x sample
    eog positions should be in brackets or an int.
    Starting from this version it does not delete channel indices after the target index.
    after eog removal x needs to be reshaped into sample x (2d) x chan x time
    :param x:
    :param y:
    :return:
    """
    # remove eog is done before reshaping
    if remove_eog_opt is True:
        x = remove_eog(x, eog_positions)

    # swap axes because the original was time x eeg x sample, while for actual learning purposes
    # sample dimension should be at the first dimension
    x = x.swapaxes(0, 2)

    # for certain deep learning frameworks data input needs to be in tensor format so we convert it to 4d here
    # but other frameworks e.g. braindecoder have their internal conversion that intrinsically does it for you.
    # this is why the default option is false.
    if to_4d is True:
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

    return x

#def balance_trials(x, y, args):
#    if


def balance_trials_2cwi(x, y, randomSample=True, sort_merged_indices=False):
    """
    Within participant version, for binary classification only
    Assumes both x and y are still numpy variables at this point, with first dimension of x being samples
    Y could either be (N) or (N,class).
    One-hot encoded labels are not supported yet.
    Balancing in this function is always done by undersampling the dominant class
    :param x:
    :param y:
    :return:
    """
    # setting random seed at every call of this function to ensure the seed remains
    # the same for all balancing acts
    np.random.seed(42)

    # we balance based on class label distribution, so first we must ensure the dimension of y var is valid
    y_dimlen = len(np.shape(y))
    assert(y_dimlen>0)

    # if it's 1 dimension then it's just (sample) but if it has 2 dimensions then its likely (sample, class)
    if y_dimlen == 1:
        # class 1
        df = np.asarray(np.where(y[:] == 0))
        # class 2
        dr = np.asarray(np.where(y[:] == 1))
    elif y_dimlen == 2:
        y = 1 - np.argmax(y, axis=0) # because it's binary and we started with remembered...
        # class 1
        # df = np.asarray(np.where(y[:,0] == 0))
        # # class 2
        # dr = np.asarray(np.where(y[:,0] == 1))
        df = np.asarray(np.where(y[:] == 0))
        # class 2
        dr = np.asarray(np.where(y[:] == 1))
    else:
        raise NameError('YDimError')

    # balancing should be done by removing from the label that has more samples than its counterpart
    # Excess trials are thrown out after random selections.
    if randomSample is True:
        if np.shape(dr)[1] > np.shape(df)[1]:
            dr = np.random.choice(dr.reshape(-1), df.shape[1], replace=False)
        else:
            df = np.random.choice(df.reshape(-1), dr.shape[1], replace=False)
    # In edge cases such as emulating on-line training with offline data,
    # data may need to be chronologically undersampled..?
    # I'd argue randomly undersampling and the re-sorting the merged indexes should be more accurate (see below)
    else:
        if np.shape(dr)[1] > np.shape(df)[1]:
            dr = dr[:df.shape[1]]
        else:
            df = df[:dr.shape[1]]

    # merge_idx represents the samples that have been selected in either class after bal
    merge_idx = np.append(dr, df)
    # if we're emulating on-line training with offline data, it may be necessary to return sorted indices
    if sort_merged_indices is True:
        merge_idx = np.sort(merge_idx)

    x = x[merge_idx]
    y = y[merge_idx]

    # returning merge_idx can be useful in session_based training where there's a separate variable for tracking
    # session indices of each trial. In other cases it can simply be ignored.
    return x, y, merge_idx

def balance_trials_mlc(x, y, randomSample=True, sort_merged_indices=False):
    np.random.seed(42)

    y_dimlen = len(np.shape(y))
    if y_dimlen == 1:
        numerical_y = y
    elif y_dimlen == 2:
        numerical_y = np.argmax(y,axis=0)
    else:
        raise NameError('y dimension does not follow rules')
    # one hot vectors must be converted into numerical for mlcs
    uniq_classes = np.unique(numerical_y)

    n_counts = []
    for uniq_class in uniq_classes:
        n_counts.append(np.where(numerical_y==uniq_class)[0].shape[0])

    min_class_idx = np.argmin(n_counts)
    min_class_n = np.min(n_counts)

    tb_merged_idxs = []


    for uniqidx, uniq_class in enumerate(uniq_classes):
        numy_indices = np.where(numerical_y==uniq_class)[0]
        if randomSample is True:
            if n_counts[uniqidx] == min_class_n:
                bal_idx = numy_indices
            elif n_counts[uniqidx] > min_class_n:
                bal_idx = np.random.choice(numy_indices, min_class_n, replace=False)
            else:
                raise NameError('n count is smaller than min n count')
        else:
            if n_counts[uniqidx] == min_class_n:
                bal_idx = numy_indices
            elif n_counts[uniqidx] > min_class_n:
                bal_idx = numy_indices[:min_class_n]
            else:
                raise NameError('n count is smaller than min n count')
        tb_merged_idxs.append(bal_idx)

    merged_idx = np.concatenate(tb_merged_idxs)
    if sort_merged_indices is True:
        merged_idx = np.sort(merged_idx)

    x = x[merged_idx]
    y = numerical_y[merged_idx]

    return x,y, merged_idx


def y_onehot_to_numerical(y):

    np.random.seed(42)

    y_dimlen = len(np.shape(y))
    if y_dimlen == 1:
        numerical_y = y
    elif y_dimlen == 2:
        numerical_y = np.argmax(y, axis=0)
    else:
        raise NameError('y dimension does not follow rules')
    # one hot vectors must be converted into numerical for mlcs

    return numerical_y

def balance_trials(x, y, randomSample=True, sort_merged_indices=False):
    ydim = len(np.shape(y))
    # rtrx, rtry, rtridx = None, None, None
    if ydim == 1:
        if np.unique(y).shape[0] > 2:
            rtrx, rtry, rtridx = balance_trials_mlc(x,y,randomSample, sort_merged_indices)
        else:
            rtrx, rtry, rtridx = balance_trials_2cwi(x, y, randomSample, sort_merged_indices)
    elif ydim == 2:
        if y.shape[0] > 2:
            rtrx, rtry, rtridx = balance_trials_mlc(x, y, randomSample, sort_merged_indices)
        else:
            rtrx, rtry, rtridx = balance_trials_2cwi(x, y, randomSample, sort_merged_indices)
    else:
        raise NameError('y dimension is not right')

    return rtrx, rtry, rtridx

def scalize_mk2_sesh(y_traindist, y_sesh, scale_punit=0.1, random_seed=42, shuffle=True):
    '''
    Creating test set and training sets (yet to be divided into n folds) for partial dataset learning
    return sample indices
    :param y_traindist:
    :param y_sesh:
    :param scale_punit:
    :param random_seed:
    :param shuffle:
    :return:
    '''
    train_stack = [] # array type
    test_stack = []  # array type
    orig_sample_size = y_traindist.shape[0]

    # setup forloop stage for changing scaling fractions
    # lock test set size to 10% of total distribution? 5% c1 and 5% c2
    # should stay the same throughout "folds" or scale iterations
    # starting trainset : 50 samples or 10% of remainder?
    # for each scale value
    dset_idx = np.arange(0, y_traindist.shape[0])

    def random_sample(base_dataset, test_fraction, seed=random_seed, shuffle_opt=shuffle, backwards=False):

        np.random.seed(seed=seed)
        sample_fraction = np.ceil(base_dataset.shape[0] * test_fraction).astype(int)
        base_dindex = np.arange(0,base_dataset.shape[0])  # base_dataset.cpu().numpy().reshape(-1)

        if shuffle_opt is True:
            sample_idx = np.random.choice(base_dindex, sample_fraction, replace=False)
        else:
            if backwards is True:
                sample_idx = base_dindex[-sample_fraction:]
            else:
                sample_idx = base_dindex[:sample_fraction]

        return sample_idx # idx within index

    uniq_sesh = np.unique(y_sesh).astype(int)
    sesh_remainders = []
    test_set = np.array([])
    remainder_set = np.array([])
    for sesh in uniq_sesh:
        sesh_idx = np.where(y_sesh==sesh)[0] # idx in actual dataset
        sesh_testidx = random_sample(sesh_idx, 0.1, shuffle_opt=False, backwards=True) #idx of idx
        #np.random.shuffle(sesh_testidx)
        sesh_testset = sesh_idx[sesh_testidx].astype(int) # idx in actual dataset
        test_set = np.append(test_set, sesh_testset).astype(int) # idx in actual dataset, multisession
        sesh_remainder = np.delete(sesh_idx, sesh_testidx) # idx in actual dataset, sesh portion
        sesh_remainders.append(sesh_remainder)

    test_set = np.sort(test_set)

    for perctage in np.arange(0.1, 1.1, scale_punit):
        perc_trainset = np.array([])
        for sesh in uniq_sesh:
            train_sample_idx_from_remainder = random_sample(sesh_remainders[np.where(uniq_sesh==sesh)[0][0]], perctage) #idx of idx
            sesh_trainset = sesh_remainders[np.where(uniq_sesh==sesh)[0][0]][train_sample_idx_from_remainder] #idx in actual dataset
            perc_trainset = np.append(perc_trainset, sesh_trainset).astype(int) # idx in atual dataset

        perc_trainset = np.sort(perc_trainset).astype(int)  # idx in actual dataset
        train_stack.append(perc_trainset)
        test_stack.append(test_set)

    return train_stack, test_stack


def label_num_to_cat(y):
    # re-aligns numerical labels to categorical for easy classification

    y_uniq = np.unique(y)
    for yidx, y_indi in np.ndenumerate(y):
        y[yidx] = np.where(y_uniq == y_indi)[0]

    return y


def export_param_json(param_dict, destination):
    """
    Assumes full param dictionary and destination folder (filename should not be included)
    destination name should include filerun identification(time-identifiable), param settings and model identification
    :param param_dict:
    :param destination:
    :return:
    """
    # if not os.path.exists(destination):
    #     os.makedirs(destination)
    with open(destination + '.json', 'w') as f:
        json.dump(param_dict, f, ensure_ascii=False, indent=2)

    return


def export_pkl(param_dict, destination):
    with open(destination+'.pkl',
              'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(param_dict, f)

    return

def proc_jumpmeans_giga(x: np.ndarray, jump_interval, time_axis=1):
    '''
    A bit weird, but in openbmi's implementation jump_interval dictates the final time length,
    instead of averaging every jump_interval
    :param x:
    :param jump_interval:
    :param time_axis:
    :return:
    '''
    new_tsamp = np.floor(x.shape[time_axis]/jump_interval).astype(int)
    newshape = list(x.shape)
    newshape[time_axis] = jump_interval # in the nongiga version it would be new_tsamp
    newx = np.zeros(newshape)

    if time_axis == 0:
        if len(x.shape) == 3:
            for timeiter in np.arange(jump_interval): # in the nongiga version it would be new_tsamp
                newx[timeiter, :, :] = np.mean(x[timeiter*new_tsamp:(timeiter+1)*new_tsamp, :, :], axis=time_axis)
        else:
            for timeiter in np.arange(jump_interval):
                newx[timeiter, :, :, :] = np.mean(x[timeiter*new_tsamp:(timeiter+1)*new_tsamp, :, :, :], axis=time_axis)
    elif time_axis ==1:
        if len(x.shape) == 3:
            for timeiter in np.arange(jump_interval):
                newx[:, timeiter, :] = np.mean(x[:, timeiter*new_tsamp:(timeiter+1)*new_tsamp, :], axis=time_axis)
        else:
            for timeiter in np.arange(jump_interval):
                newx[:, timeiter, :, :] = np.mean(x[:, timeiter*new_tsamp:(timeiter+1)*new_tsamp, :, :], axis=time_axis)
    elif time_axis==2:
        if len(x.shape) == 3:
            for timeiter in np.arange(jump_interval):
                newx[:, :, timeiter] = np.mean(x[:,:, timeiter*new_tsamp:(timeiter+1)*new_tsamp], axis=time_axis)
        else:
            for timeiter in np.arange(jump_interval):
                newx[:,:,timeiter, :] = np.mean(x[:,:, timeiter*new_tsamp:(timeiter+1)*new_tsamp, :], axis=time_axis)
    elif time_axis==3:
            for timeiter in np.arange(jump_interval):
                newx[:, :, :, timeiter] = np.mean(x[:,:,:, timeiter*new_tsamp:(timeiter+1)*new_tsamp], axis=time_axis)

    else:
        raise NameError('wrong time axis')
        return

    return newx

def proc_jumpmeans(x: np.ndarray, jump_interval, time_axis=1):
    '''
    A bit weird, but in openbmi's implementation jump_interval dictates the final time length,
    instead of averaging every jump_interval
    :param x:
    :param jump_interval:
    :param time_axis:
    :return:
    '''
    new_tsamp = np.floor(x.shape[time_axis]/jump_interval).astype(int)
    newshape = list(x.shape)
    newshape[time_axis] = new_tsamp# in the nongiga version it would be new_tsamp
    newx = np.zeros(newshape)

    if time_axis == 0:
        if len(x.shape) == 3:
            for timeiter in np.arange(new_tsamp): # in the nongiga version it would be new_tsamp
                newx[timeiter, :, :] = np.mean(x[timeiter*jump_interval:(timeiter+1)*jump_interval, :, :], axis=time_axis)
        else:
            for timeiter in np.arange(new_tsamp):
                newx[timeiter, :, :, :] = np.mean(x[timeiter*jump_interval:(timeiter+1)*jump_interval, :, :, :], axis=time_axis)
    elif time_axis ==1:
        if len(x.shape) == 3:
            for timeiter in np.arange(new_tsamp):
                newx[:, timeiter, :] = np.mean(x[:, timeiter*jump_interval:(timeiter+1)*jump_interval, :], axis=time_axis)
        else:
            for timeiter in np.arange(new_tsamp):
                newx[:, timeiter, :, :] = np.mean(x[:, timeiter*jump_interval:(timeiter+1)*jump_interval, :, :], axis=time_axis)
    elif time_axis==2:
        if len(x.shape) == 3:
            for timeiter in np.arange(new_tsamp):
                newx[:, :, timeiter] = np.mean(x[:,:, timeiter*jump_interval:(timeiter+1)*jump_interval], axis=time_axis)
        else:
            for timeiter in np.arange(new_tsamp):
                newx[:,:,timeiter, :] = np.mean(x[:,:, timeiter*jump_interval:(timeiter+1)*jump_interval, :], axis=time_axis)
    elif time_axis==3:
            for timeiter in np.arange(new_tsamp):
                newx[:, :, :, timeiter] = np.mean(x[:,:,:, timeiter*jump_interval:(timeiter+1)*jump_interval], axis=time_axis)

    else:
        raise NameError('wrong time axis')
        return

    return newx