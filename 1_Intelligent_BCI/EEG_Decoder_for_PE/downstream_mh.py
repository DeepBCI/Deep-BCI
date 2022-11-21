import torch
import tqdm
import argparse
import objgraph
import time
import utils_mh
from result_tracking_mh import ThinkerwiseResultTracker

from dn3_configuraotron_config_mh import ExperimentConfig
from dn3_data_dataset_mh import Thinker
from processes_mh import StandardClassification
from dn3_ext_mh import BENDRClassification, LinearHeadBENDR, OnlyCNN

from datetime import datetime
import os
time_now = datetime.now().strftime('%Y%m%d_%H%M')

from pytorchtools_mh import EarlyStopping

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)

from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True) # TODO
# 시간_model -> 시간_model_metric
# 데이터셋 > model > fold -> 데이터셋 > model > metric > fold

if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils_mh.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="configs/downstream_mh.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="configs/metrics_mh.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    # parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
    #                                                              'final results.')
    parser.add_argument('--results-filename', default='./results/' + time_now + '_' + parser.parse_args().model + '.xlsx', help='What to name the spreadsheet produced with all '
                                                                 'final results.') # parser.parse_args().model
    args = parser.parse_args()

    experiment = ExperimentConfig(args.ds_config)
    if args.results_filename:
        results = ThinkerwiseResultTracker()

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
        added_metrics, retain_best, _ = utils_mh.get_ds_added_metrics(ds_name, args.metrics_config)
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils_mh.get_lmoso_iterator(ds_name, ds))):
            # if fold < 28 :
            #     continue # TODO
            tqdm.tqdm.write(torch.cuda.memory_summary())
            writer = SummaryWriter()
            flg = False
            if args.model == utils_mh.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu)
                flg = True
            elif args.model == utils_mh.MODEL_CHOICES[1]:
                model = LinearHeadBENDR.from_dataset(training)
                flg = True
            elif args.model == utils_mh.MODEL_CHOICES[2]:
                model = OnlyCNN.from_dataset(training)

            # #TODO 221109
            # flg = False
            # if not args.random_init and flg:
            #     # model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
            #     #                               freeze_encoder=True, freeze_contextualizer=True)
            #     model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
            #                                   freeze_encoder=args.freeze_encoder)

            # model = LittleEEGNet.from_dataset(training)
            model.to(device)
            process = StandardClassification(model, metrics=added_metrics, writer = writer) # TODO
            process.set_optimizer(torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01)) # weight_decay == L2 loss


            # def epoch_checkpoint(metrics):  # ds_name, fold, args.model
            #     tqdm.tqdm.write("Saving...") # 데이터셋 > model > fold -> 데이터셋 > model > metric > fold
            #     path = 'checkpoints/epoch' #/{0}/{1}_{2}/{3}_fold'.format(ds_name, args.model, retain_best, fold)
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     model.save(path + '/model_epoch_{}.pt'.format(
            #         metrics['epoch']))  # TODO 돌린 데이터셋, lmso(fold), 모델 이름...  dataset > model > lmss

            def simple_checkpoint(metrics):  # ds_name, fold, args.model
                tqdm.tqdm.write("Saving...")
                if fold +1 >= 19:
                    name_fold = fold + 2
                else:
                    name_fold = fold

                path = 'checkpoints/{0}/{1}_{2}/{3}_fold'.format(ds_name, args.model, retain_best, fold+1)
                if not os.path.exists(path):
                    os.makedirs(path)
                model.save(path + '/model_check_fold{0}.pt'.format(fold+1))


            simple_checkpoint(None)
            # Fit everything
            # process.fit(training_dataset=training, validation_dataset=validation, epoch_callback=epoch_checkpoint,
            #             log_callback=simple_checkpoint, warmup_frac=0.1,
            #             retain_best=retain_best, pin_memory=False, **ds.train_params)   step_callback= [EarlyStopping(monitor = , mode = "max")],
            # TODO 221026
            # early_stopping = EarlyStopping(model, path = 'checkpoints/{0}/{1}_{2}/{3}_fold'.format(ds_name, args.model, retain_best, fold+1) + '/model_check_fold{0}_earlystop.pt'.format(fold+1))

            # process.fit(training_dataset=training, validation_dataset=validation,
            #             log_callback=simple_checkpoint, warmup_frac=0.1, epoch_callback= early_stopping,
            #             retain_best=retain_best, pin_memory=False, **ds.train_params)

            a, b = process.fit(training_dataset=training, validation_dataset=validation,
                        log_callback=simple_checkpoint, warmup_frac=0.1,
                        retain_best=retain_best, pin_memory=False, **ds.train_params)
            print(a)
            print("--")
            print(b)
            if args.results_filename:
                if isinstance(test, Thinker):
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                file_name = './results/' + time_now + '_' + parser.parse_args().model + '_{}.xlsx'.format(retain_best)
                results.to_spreadsheet(file_name) #args.results_filename) # './results/' + parser.parse_args().model + time_now + '.xlsx' ;  dataset > model > lmso == dataset_name, model_name, fold
                # 시간_model -> 시간_model_metric
            # tqdm.tqdm.write("Saving best model...")
            # model.save('checkpoints/model_best_val.pt')

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            objgraph.show_backrefs(model, filename='sample-backref-graph.png')
            del model
            torch.cuda.synchronize()
            time.sleep(10)

            writer.flush()
            writer.close()



        if args.results_filename:
            results.performance_summary(ds_name) # 시간_model -> 시간_model_metric
            file_name = './results/' + time_now + '_' + parser.parse_args().model + '_{}.xlsx'.format(retain_best)
            results.to_spreadsheet(file_name)
            # results.to_spreadsheet(args.results_filename) #'./results/' + time_now + '_' + parser.parse_args().model + '_summary' + '.xlsx') #args.results_filename) # './results/' + time_now + '_' + parser.parse_args().model + '_summary' + '.xlsx'
