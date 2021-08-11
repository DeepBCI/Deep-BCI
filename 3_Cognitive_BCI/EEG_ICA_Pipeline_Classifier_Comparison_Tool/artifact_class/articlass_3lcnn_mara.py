from .articlass_helper  import *
from .trainfunc_3lcnn import *
from .architectures import artifact_class
from .io_handler.pytorch_output_articomp import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import argparse
import yaml
import logging as pylog
from datetime import datetime
import json
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

""" This is an example script file for running multiple preprocessed pipelines of the same dataset with a CNN.
    You will be designating detailed configurations (dataset pipeline name, source, balancing strategy, training and netowkr parameters)
    in a separate .yaml file in the configs folder. """

def get_cmdarg():
    # using argparse for parsing arguments
    # configs tend to vary a lot by experiments so let's use yaml to get around possible headaches
    psr = argparse.ArgumentParser()
    # psr.add_argument()

    """
    remember that typechecking is not always necessary!
    example: psr.add_argument("--n", default=0, help='helptext')

    """

    psr.add_argument("--run_type", default='arti vs no artifact performance', help='helptext')
    psr.add_argument("-dataset_config", default="../configs/dataset_configs/icacomp/mem_askrw_loss_nd.yaml",
                     help="config containing datapath, x and label options. yaml format")
    psr.add_argument("-network_config", default="../configs/network_configs/icacomp/3lcnn.yaml",
                     help="config containing network parameters. yaml format")
    psr.add_argument("-train_config", default="../configs/train_configs/icacomp/3lcnn.yaml",
                     help="config containing general training parameters such as batch size. yaml format")

    opt, _ = psr.parse_known_args()
    return opt


# process running arguments
runopt = get_cmdarg()

# process dataset config first
with open(runopt.dataset_config, 'r') as stream:
    try:
        dataset_config = yaml.load(stream, Loader=yaml.FullLoader)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)
# network configs
with open(runopt.network_config, 'r') as stream:
    try:
        network_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
# train configs
with open(runopt.train_config, 'r') as stream:
    try:
        train_config = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

root_dir = 'D:\\data/bbciMat/cogsys_transfer_lie/'

# set name for log file
run_init_time = datetime.now()
output_dir = './run_logs/'
output_prefix = '{0}_{1}'.format(os.path.basename(sys.argv[0]), run_init_time.strftime('%y%m%d_%H%M%S'))
log_fname = '{0}{1}'.format(output_dir, output_prefix)
pylog.basicConfig(filename=log_fname, level=pylog.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
pylog.debug('Begin debugging.')

# log config details (because the same config file may change throughout life)
pylog.info('dataset configs::{}'.format(json.dumps(dataset_config)))
pylog.info('network configs::{}'.format(json.dumps(network_config)))
pylog.info('training configs::{}'.format(json.dumps(train_config)))

# data loading design: load once per particiapnt
# load one file for prototyping
# dataset_infomat = load_matfile(root_dir + 'train_ready_noica/datasetinfo.mat')
# dataset_sbj1 = load_matfile(root_dir + 'train_ready_noica/sbj_1.mat')

if dataset_config['dataset_info_exists'] == 1:
    pylog.info("dataset representative infomat exists")
    ica_datinfo = load_matfile(dataset_config['datapaths']['mara']+'datasetinfo')
    #noica_datinfo = load_matfile(dataset_config['datapaths']['noica']+'datasetinfo')
    ica_trialsbj = ica_datinfo['trial_sbj']
    #noica_trialsbj = noica_datinfo['trial_sbj']

else:
    pylog.info("dataset representative infomat doesn't exist")

# init network parameters
filt_sizes = network_config['filt_sizes']
kernels_3l = network_config['kernels']
strides_3l = network_config['strides']
dilations_3l = network_config['dilations']
padds_3l = network_config['paddings']

# training parameters
patience_threshold = train_config['early_stop_patience']

# set torch device related settings
device = torch.device(train_config['device'] if torch.cuda.is_available() else "cpu")
model_ica = None
model_noica = None
torch.manual_seed(42)
torch.backends.cudnn.deterministic = train_config['determinism']
torch.backends.cudnn.benchmark = train_config['determinism_bench']


# could be participant, group, whatever divisions you have for dataset files
# ica and nonica should have identical number of subgroups given a task
# assuming in this version that all datasets files are separated by participant,
# prepping of data should be done within load loop

# init metric containers - group
#runmet_ica = MetricAggregates(container_type='mara')
#runmet_noica = MetricAggregates(container_type='noica')


#set_to_run = ['abl_10']
set_to_run = ['ica',]


for runset in  set_to_run:

    runmet_runset = MetricAggregates(container_type=runset)


    for sbjidx in np.arange(dataset_config['participants_in_data']):
        pylog.debug("processing partiicpant {}".format(sbjidx))
        # load both datasets now because if we're loading datasets individually they're supposed to be pretty small
        sbjidx_nat = sbjidx + 1
        #mat_noica = load_matfile(dataset_config['datapaths']['noica'] + dataset_config['datafiles_prefix'] + str(sbjidx_nat) + '.mat')
        mat_ica = load_matfile(dataset_config['datapaths'][runset] + dataset_config['datafiles_prefix'] + str(sbjidx_nat) + '.mat')

        sbjmet_ica = MetricAggregates(container_type='individual', member_type='fold')
        #sbjmet_noica = MetricAggregates(container_type='individual', member_type='fold')

        # preproccing x data. in 3L they need to be in tensor form so we do that too
        x_ica = preproc_x(mat_ica[dataset_config['data_to_use']], remove_eog_opt=True,
                          eog_positions=dataset_config['eog_channels'], to_4d=True)

        # x_noica = preproc_x(mat_noica[dataset_config['data_to_use']], remove_eog_opt=True,
        #                     eog_positions=dataset_config['eog_channels'], to_4d=True)

        # length of time dimension for network definition later
        time_len = x_ica.shape[-1]

        # labels. y data is in class x sample.
        # pytorch doesn't need em to be in one hot format so convert that
        # assuming it's binary this should be ezpz
        # if statement is lazily evaluated so it wouldn't matter even if trim data doesn't exists for the latter condition
        if 'trim_data' in dataset_config and dataset_config['trim_data'] is True:
            # trim data is only binary for now. If we support multiclass on top of this this part needs
            # revision
            c1_idx = np.where(mat_ica[dataset_config['label_to_use']][0, :] == 1)[0]
            c2_idx = np.where(mat_ica[dataset_config['label_to_use']][1, :] == 1)[0]
            x_ica, y_ica = trim_trials(x_ica, mat_ica[dataset_config['label_to_use']], np.concatenate([c1_idx, c2_idx]))
            y_ica = y_ica[0, :].reshape(-1)
            n_before_bal_ica = y_ica.shape[0]

            # c1_idx = np.where(mat_noica[dataset_config['label_to_use']][0, :] == 1)[0]
            # c2_idx = np.where(mat_noica[dataset_config['label_to_use']][1, :] == 1)[0]
            # x_noica, y_noica = trim_trials(x_noica, mat_noica[dataset_config['label_to_use']],
            #                                np.concatenate([c1_idx, c2_idx]))
            # y_noica = y_noica[0, :].reshape(-1)
        else:
            y_ica = mat_ica[dataset_config['label_to_use']]#[0, :].reshape(-1)
            n_before_bal_ica = y_ica.shape[1]
            #y_noica = mat_noica[dataset_config['label_to_use']]#[0, :].reshape(-1)



        # n_before_bal_noica = y_noica.shape[0]

        # balancing dataset if the option says so
        if dataset_config['balance_dataset'] is True:
            x_ica, y_ica, sampidx_ica = balance_trials(x_ica, y_ica, randomSample=True,
                                                            sort_merged_indices=False)
            # x_noica, y_noica, sampidx_noica = balance_trials(x_noica, y_noica, randomSample=True,
            #                                                       sort_merged_indices=False)

            pylog.debug('bal. {0} dataset class 0: {1}, class 1: {2}'.format(runset, y_ica[y_ica == 0].shape[0],
                                                                         y_ica[y_ica == 1].shape[0]))
            # pylog.debug('bal.nonica dataset class 0: {0}, class 1: {1}'.format(y_noica[y_noica == 0].shape[0],
            #                                                                y_noica[y_noica == 1].shape[0]))
            n_after_bal_ica = y_ica.shape[0]
            # n_after_bal_noica = y_noica.shape[0]

        # are _fold_split variables reusable? Testing needed on this
        tt_fold = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        tv_fold = StratifiedKFold(n_splits=train_config['folds'], shuffle=True, random_state=42)

        # work on MARA datasets
        # don't really feel like turning this part into a function
        pylog.debug("splitting folds - {}".format(runset))
        # holding out 10% as test set before train and validation set creation
        for tv_idx, test_idx in tt_fold.split(x_ica, y_ica):
            x_tv, y_tv = x_ica[tv_idx], y_ica[tv_idx]
            x_test, y_test = x_ica[test_idx], y_ica[test_idx]

        test_set = TensorDataset(torch.from_numpy(x_test).float().to(device),
                                 torch.from_numpy(y_test).long().reshape(-1,1).to(device))
        test_loader = DataLoader(test_set, batch_size=train_config['test_batch'], shuffle=True)

        # splitting folds
        fold_idx = 0
        for train_idx, val_idx in tv_fold.split(x_tv, y_tv):
            fold_idx += 1
            x_train, y_train = x_tv[train_idx], y_tv[train_idx]
            x_val, y_val = x_tv[val_idx], y_tv[val_idx]

            #init fold metric aggregators
            foldmet_ica = MetricAggregates(container_type='fold')

            # if we have augmentations this would be where we insert them
            #

            # transforming data in to pytorch dataset/loaders
            train_set = TensorDataset(torch.from_numpy(x_train).float().to(device),
                                      torch.from_numpy(y_train).long().reshape(-1,1).to(device))
            validation_set = TensorDataset(torch.from_numpy(x_val).float().to(device),
                                           torch.from_numpy(y_val).long().reshape(-1,1).to(device))

            train_loader = DataLoader(train_set, batch_size=train_config['train_batch'], shuffle=True)
            validation_loader = DataLoader(validation_set, batch_size=train_config['test_batch'], shuffle=True)



            # init train model
            # architecture call could be transformed into a callable?
            if 'model_ica' in vars():
                del model_ica
            # if 'model_noica' in vars():
            #     del model_noica
            torch.cuda.empty_cache()
            model_ica = artifact_class.CNN3LCustomParam(convlayer_filtsizes=filt_sizes,
                                                        eeg_dim=x_train.shape[2:4],
                                                        kernels=kernels_3l,
                                                        strides=strides_3l,
                                                        dilations=dilations_3l,
                                                        padds=padds_3l,
                                                        out_classes=dataset_config['output_classn']).to(device)
            optimizer = torch.optim.Adam(model_ica.parameters(), lr=train_config['learning_rate'])

            # reset patientce before starting epochs
            current_patience = 0  # patience stacks up when no increase in performance is observed
            previous_valacc = 0  # validation metric of previous epoch

            # running across epochs
            # if we're running ensembles something should be done inside this loop
            for epo in np.arange(train_config['epochs']):
                # init MetricContainer for each epoch
                metric_cont = MetricContainer(container_type='epoch')

                train_loss, train_acc = deep_train(model=model_ica, device=device, train_loader=train_loader,
                                                   optimizer=optimizer, epoch=epo)
                val_loss, val_acc, val_target, val_pred = deep_test(epo, model=model_ica, device=device,
                                                                    test_loader=validation_loader)

                metric_cont.add_container_info({'epoch_num': epo})
                metric_cont.add_metrics({'train_loss': train_loss, 'train_acc': train_acc,
                                              'val_loss': val_loss, 'val_acc': val_acc})
                foldmet_ica.add_members(metric_cont)

                if val_acc > previous_valacc:
                    current_patience = 0
                    # prep patience thresholding for next epoch
                    # only update this if a new record is reached
                    previous_valacc = val_acc
                else:
                    current_patience += 1

                if current_patience == patience_threshold or epo+1 == train_config['epochs']:
                    # enough epochs have passed since the last accuracy increase.
                    # Or, we're at the last epoch so it's time to finish up.
                    # put in best epoch value to metric aggregator and save model.
                    # if current_patience == patience_threshold:

                    # output name is determined here because it takse epoch value into account.
                    # output filename format:
                    # experiment name(e.g. articlass)/dataset(e.g. lie)_label(e.g. lt)/model(e.g. 3lcnn_)_sbj_fold_epo
                    model_save_path = './learn_outs/weights/{0}/{1}_{2}_{3}/{4}_{5}_{6}_{7}'.format(
                        'articlass', dataset_config['dataset_name'], dataset_config['label_to_use'],
                        runset,
                        network_config['network_name'], sbjidx, fold_idx, epo
                    )


                    # test is need to be done during last and best epochs
                    test_loss, test_acc, test_target, test_pred = deep_test(epo, model=model_ica, device=device,
                                                                            test_loader=test_loader)
                    test_target = np.concatenate(test_target)
                    test_pred = np.concatenate(test_pred)

                    if epo+1 == train_config['epochs'] and current_patience <= patience_threshold:
                        # last epoch and the threshold was not reached (last epoch is best epoch)
                        foldmet_ica.add_container_info({
                            'last_epo_weight': model_save_path,
                            'best_epo_weight': model_save_path,
                            'best_epo': epo,
                        })
                        foldmet_ica.add_metrics({
                            'best_epo_test_acc': test_acc,
                            'last_epo_test_acc': test_acc,
                            'best_epo_test_loss': test_loss,
                            'last_epo_test_loss': test_loss,

                            'best_epo_val_acc': val_acc,
                            'best_epo_val_loss': val_loss,
                            'last_epo_val_acc': val_acc,
                            'last_epo_val_loss': val_loss,

                            'best_epo_train_acc': train_acc,
                            'best_epo_train_loss': train_loss,
                            'last_epo_train_acc': train_acc,
                            'last_epo_train_loss': train_loss,

                            'best_epo_test_pred': test_pred,
                            'best_epo_test_target': test_target,
                            'last_epo_test_pred': test_pred,
                            'last_epo_test_target': test_target,
                        })

                    elif current_patience == patience_threshold:
                        # threshold has reached and this is not the last epoch (best epoch)
                        foldmet_ica.add_container_info({
                            'best_epo_weight': model_save_path,
                            'best_epo': epo

                        })
                        foldmet_ica.add_metrics({
                            'best_epo_test_acc': test_acc,
                            'best_epo_test_loss': test_loss,

                            'best_epo_val_acc': val_acc,
                            'best_epo_val_loss': val_loss,

                            'best_epo_train_acc': train_acc,
                            'best_epo_train_loss': train_loss,

                            'best_epo_test_pred': test_pred,
                            'best_epo_test_target': test_target,
                        })
                    else:
                        # threshold has reached and this is the last epoch (last epoch)
                        # model_save_path = './learn_outs/weights/'
                        foldmet_ica.add_container_info({
                            'last_epo_weight': model_save_path
                        })
                        foldmet_ica.add_metrics({
                            'last_epo_test_acc': test_acc,
                            'last_epo_test_loss': test_loss,

                            'last_epo_val_acc': val_acc,
                            'last_epo_val_loss': val_loss,

                            'last_epo_train_acc': train_acc,
                            'last_epo_train_loss': train_loss,

                            'last_epo_test_pred': test_pred,
                            'last_epo_test_target': test_target,
                        })

                    sub_path = os.path.dirname(model_save_path)
                    setup_directory(sub_path)
                    torch.save({
                        'epoch': epo,
                        'model_state_dict': model_ica.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, model_save_path)

            # fold level
            # insert fold agg into sbj agg
            sbjmet_ica.add_members(foldmet_ica)

        # sbj level
        sbjmet_ica.add_container_info({
            'sbjidx': sbjidx, 'sbjidx_nat':sbjidx_nat, 'n_before_bal': n_before_bal_ica,
            'n_after_bal': n_after_bal_ica
        })
        #sbjmet_ica.member_statistics()
        # sbj leve things are done, insert sbj agg into experiment agg
        runmet_runset.add_members(sbjmet_ica)


    runmet_runset.add_container_info({'dataset_config': dataset_config,
                                   'network_config': network_config,
                                   'train_config': train_config})
    # runmet_noica.add_container_info({'dataset_config': dataset_config,
    #                                'network_config': network_config,
    #                                'train_config': train_config})


    metica_path = './learn_outs/metrics/{0}_{1}_{2}_{3}_{4}'.format(
                        'articlass', dataset_config['dataset_name'], dataset_config['label_to_use'],
                        runset,
                        network_config['network_name'],
                    )
    sub_path = os.path.dirname(metica_path)
    setup_directory(sub_path)
    export_pkl(runmet_runset, metica_path)

    #export_pkl(runmet_noica, metnoica_path)











