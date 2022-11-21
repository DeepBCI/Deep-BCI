import torch
import yaml
from dn3_ext_mh import BENDRClassification, LinearHeadBENDR
from dn3_metrics_base_mh import balanced_accuracy, auroc
from transforms_instance_mh import To1020

EXTRA_METRICS = dict(bac=balanced_accuracy,
                     auroc=auroc)

MODEL_CHOICES = ['BENDR', 'linear', 'CNN']


def make_model(args, experiment, dataset):
    if args.model == MODEL_CHOICES[0]:
        model = BENDRClassification.from_dataset(dataset)
    else:
        model = LinearHeadBENDR.from_dataset(dataset)

    if not args.random_init:
        model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                      freeze_encoder=args.freeze_encoder)

    return model


def get_ds_added_metrics(ds_name, metrics_config):
    """
    Given the name of a dataset, and name of metrics config file, returns all additional metrics needed,
    the metric to retain the best validation instance of and the chance-level threshold of this metric.
    """
    metrics = dict()
    retain_best = 'Accuracy'
    chance_level = 0.5

    with open(metrics_config, 'r') as f:
        conf = yaml.safe_load(f)
        if ds_name in conf:
            metrics = conf[ds_name]
            if isinstance(metrics[0], dict):
                metrics[0], chance_level = list(metrics[0].items())[0]
            retain_best = metrics[0]

    return {m: EXTRA_METRICS[m] for m in metrics if m != 'Accuracy'}, retain_best, chance_level


def get_ds(name, ds):
    dataset = ds.auto_construct_dataset() # 아직 90 channel
    dataset.add_transform(To1020())
    return dataset


def get_lmoso_iterator(name, ds):
    dataset = get_ds(name, ds) # channel 20..
    try:
        ds.train_params.pop('_d')
    except:
        pass
    specific_test = ds.test_subjects if hasattr(ds, 'test_subjects') else None
    iterator = dataset.lmso(ds.folds, test_splits=specific_test) \
        if hasattr(ds, 'folds') else dataset.loso(test_person_id=specific_test)
    return iterator

