import re
from sys import gettrace
from layers_mh import _make_mask, _make_span_from_seeds
from model_mh import Classifier
import tqdm.auto as tqdm
import torch
import numpy as np
from pandas import DataFrame
from collections import OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler
from transform_batch_mh import BatchTransform
from dn3_utils_mh import LabelSmoothedCrossEntropyLoss
import random
import os # 221118
# for control the random seed
random_seed = 2022
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


class BaseProcess(object):
    """
    By default, uses the SGD with momentum optimization.
    """

    def __init__(self, lr=0.001, metrics=None, evaluation_only_metrics=None, l2_weight_decay=0.00001, cuda=None, **kwargs): # TODO: l2_weight_decay = 0.01 이었음
        """
        Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
        this base class.

        By default, uses the SGD with momentum optimization.

        Parameters
        ----------
        cuda : bool, string, None
               If boolean, sets whether to enable training on the GPU, if a string, specifies can be used to specify
               which device to use. If None (default) figures it out automatically.
        lr : float
             The learning rate to use, this will probably something that should be tuned for each application.
             Start with multiplying or dividing by values of 2, 5 or 10 to seek out a good number.
        metrics : dict, list
                  A dictionary of named (keys) metrics (values) or some iterable set of metrics that will be identified
                  by their class names.
        evaluation_only_metrics : list
                                 A list of names of metrics that will be used for evaluation only (not calculated or
                                 reported during training steps).
        l2_weight_decay : float
                          One of the simplest and most common regularizing techniques. If you find a model rapidly
                          reaching high training accuracy (and not validation) increase this. If having trouble fitting
                          the training data, decrease this.
        kwargs : dict
                 Arguments that will be used by the processes' :py:meth:`BaseProcess.build_network()` method.
        """
        if cuda is None:
            cuda = torch.cuda.is_available()
            if cuda:
                tqdm.tqdm.write("GPU(s) detected: training and model execution will be performed on GPU.")
        if isinstance(cuda, bool):
            cuda = "cuda" if cuda else "cpu"
        assert isinstance(cuda, str)
        self.cuda = cuda
        self.device = torch.device(cuda)
        self._eval_metrics = list() if evaluation_only_metrics is None else list(evaluation_only_metrics).copy()
        self.metrics = OrderedDict()
        if metrics is not None:
            if isinstance(metrics, (list, tuple)):
                metrics = {m.__class__.__name__: m for m in metrics}
            if isinstance(metrics, dict):
                self.add_metrics(metrics)

        _before_members = set(self.__dict__.keys())
        self.build_network(**kwargs)
        new_members = set(self.__dict__.keys()).difference(_before_members)
        self._training = False
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
                    self._trainables.append(member)

                self.__dict__[member] = self.__dict__[member].to(self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), weight_decay=l2_weight_decay, lr=lr, nesterov=True,
                                         momentum=0.9) # 변경 됨 ㅇㅇ
        self.scheduler = None # pytorch에서 제공하는 Learning rate scheduler
        self.scheduler_after_batch = False
        self.epoch = None
        self.lr = lr
        self.weight_decay = l2_weight_decay

        self._batch_transforms = list()
        self._eval_transforms = list()

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        del self.optimizer
        self.optimizer = optimizer
        self.lr = float(self.optimizer.param_groups[0]['lr']) # lr의 변화 찍어볼 수 있음

    def set_scheduler(self, scheduler, step_every_batch=False):
        """
        This allow the addition of a learning rate schedule to the process. By default, a linear warmup with cosine
        decay will be used. Any scheduler that is an instance of :any:`Scheduler` (pytorch's schedulers, or extensions
        thereof) can be set here. Additionally, a string keywords can be used including:
          - "constant"

        Parameters
        ----------
        scheduler: str, Scheduler
        step_every_batch: bool
                          Whether to call step after every batch (if `True`), or after every epoch (`False`)

        """
        if isinstance(scheduler, str):
            if scheduler.lower() == 'constant':
                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: 1.0)
            else:
                raise ValueError("Scheduler {} is not supported.".format(scheduler))
        # This is the most common one that needs this, force this to be true
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler_after_batch = True
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            self.scheduler_after_batch = True # TODO 221006 : lr 스케쥴러 종류 바꿈
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            self.scheduler_after_batch = True # TODO 221006 : lr 스케쥴러 종류 바꿈
        else:
            self.scheduler_after_batch = step_every_batch
        self.scheduler = scheduler

    def add_metrics(self, metrics: dict, evaluation_only=False):
        self.metrics.update(**metrics)
        if evaluation_only:
            self._eval_metrics += list(metrics.keys())

    def _optimize_dataloader_kwargs(self, num_worker_cap=6, **loader_kwargs):
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        # Use multiple worker processes when NOT DEBUGGING
        if gettrace() is None:
            try:
                # Find number of cpus available (taken from second answer):
                # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
                m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                              open('/proc/self/status').read())
                nw = bin(int(m.group(1).replace(',', ''), 16)).count('1')
                # Cap the number of workers at 6 (actually 4) to avoid pummeling disks too hard
                nw = min(num_worker_cap, nw)
            except FileNotFoundError:
                # Fallback for when proc/self/status does not exist
                nw = 2
        else:
            # 0 workers means not extra processes are spun up
            nw = 2
        # loader_kwargs.setdefault('num_workers', int(nw - 2))
        # print("Loading data with {} additional workers".format(loader_kwargs['num_workers']))
        return loader_kwargs

    def _get_batch(self, iterator):
        batch = [x.to(self.device, non_blocking=self.cuda == 'cuda') for x in next(iterator)]
        xforms = self._batch_transforms if self._training else self._eval_transforms
        for xform in xforms:
            if xform.only_trial_data:
                batch[0] = xform(batch[0])
            else:
                batch = xform(batch)
        return batch

    def add_batch_transform(self, transform: BatchTransform, training_only=True):
        self._batch_transforms.append(transform)
        if not training_only:
            self._eval_transforms.append(transform)

    def clear_batch_transforms(self):
        self._batch_transforms = list()
        self._eval_transforms = list()

    def build_network(self, **kwargs):
        """
        This method is used to add trainable modules to the process. Rather than placing objects for training
        in the __init__ method, they should be placed here.

        By default any arguments that propagate unused from __init__ are included here.
        """
        self.__dict__.update(**kwargs)

    def parameters(self):
        """
        All the trainable parameters in the Trainable. This includes any architecture parameters and meta-parameters.

        Returns
        -------
        params :
                 An iterator of parameters
        """
        for member in self._trainables:
            yield from self.__dict__[member].parameters()

    def forward(self, *inputs):
        """
        Given a batch of inputs, return the outputs produced by the trainable module.

        Parameters
        ----------
        inputs :
               Tensors needed for underlying module.

        Returns
        -------
        outputs :
                Outputs of module

        """
        raise NotImplementedError

    def calculate_loss(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the loss.

        Returns
        -------
        Loss :
             Single loss quantity to be minimized.
        """
        if isinstance(outputs, (tuple, list)):
            device = outputs[0].device
        else:
            device = outputs.device
        loss_fn = self.loss

        if hasattr(self.loss, 'to'):
            loss_fn = loss_fn.to(device)

        # TODO : loss function BCE로.. 221115
        # input_ohe = torch.nn.functional.one_hot(inputs[-1], 2).float()
        # l_temp = loss_fn(outputs.squeeze(-1).float(), inputs[-1].float())
        l_temp = loss_fn(outputs, inputs[-1])
        if torch.any(torch.isnan(l_temp)):
        # if torch.any(torch.isnan(loss_fn(outputs, inputs[-1]))): # outputs -> nan....?
            print("nan_222") # TODO 220914 first error - metrics O 220922
            l_nonan = loss_fn(outputs, inputs[-1])
            # l_nonan[l_nonan != l_nonan] = float("Inf")
            # l_nonan[l_nonan != l_nonan] = float(0) # grad_fn : NllLossBackward -> CopySlices
            l_nonan = torch.rand_like(l_nonan).requires_grad_() # 필요한듯
            return l_nonan

        return l_temp
        # return loss_fn(outputs, inputs[-1]) # outputs == input: (minibatches, class) = (60, 2) ||  inputs[-1] == target (minibatch, )
        # pytorch에서는 softmax와 cross-entropy합쳐놔서 dog, cat = (0.1, 0.9) - target 1  이렇게 매칭.
        # 잘 하고 있음..

    def calculate_metrics(self, inputs, outputs):
        """
        Given the inputs to and outputs from the underlying module. Return tracked metrics.

        Parameters
        ----------
        inputs :
               Input tensors.
        outputs :
                Output tensors.

        Returns
        -------
        metrics : OrderedDict, None
                  Dictionary of metric quantities.
        """
        metrics = OrderedDict()
        for met_name, met_fn in self.metrics.items():
            if self._training and met_name in self._eval_metrics:
                continue
                # pass # TODO 220914 220922
            try:
                metrics[met_name] = met_fn(inputs, outputs)
            # I know its super broad, but basically if metrics fail during training, I want to just ignore them...
            except:
                continue
        return metrics

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)

    def train_step(self, *inputs) : #, epoch): # TODO 221021
        self.train(True)
        outputs = self.forward(*inputs) #, epoch = epoch) # TODO 221018 TODO 221021 221021
        # if torch.any(torch.isnan(outputs)): # TODO 221114
        #     print("nan_280")
        #     # outputs_ = outputs
        #     # outputs_[outputs_ != outputs_] = float("Inf")
        #     # outputs_[outputs_ != outputs_] = float(0)
        #     outputs = torch.rand_like(outputs).requires_grad_() # TODO 220926 - 필요
            # outputs = outputs_
        # TODO self.loss 정의 됐는지 확인, binary -> target=target.unsqueeze(1)
        # if self.loss == torch.nn.BCEWithLogitsLoss():
        #     inputs[-1] = inputs[-1].unsqueeze(1)
        loss = self.calculate_loss(inputs, outputs) # inputs = 2 by 60 by 20 by 896, outputs = 60 by 2
        # self.writer.add_scalar("loss_train", loss, self.epoch)
        self.backward(loss)
        # print(loss) # TODO 221115
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 5) # TODO 220926
        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_after_batch:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs)
        train_metrics.setdefault('loss', loss.item())

        # metric_name_for_writer = list(train_metrics.keys())
        # if self._training :
        #     for name_i in metric_name_for_writer :
        #         self.writer.add_scalar(name_i+"_train", train_metrics[name_i], self.epoch) # TODO 220929 # TODO 221004
        #         # self.writer.add_scalar(name_i + "_train", metrics[name_i], self.epoch)

        return train_metrics

    def evaluate(self, dataset, **loader_kwargs):
        """
        Calculate and return metrics for a dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        metrics : OrderedDict
                Metric scores for the entire
        """
        self.train(False)
        inputs, outputs = self.predict(dataset, **loader_kwargs) # TODO check output

        # if torch.any(torch.isnan(outputs)): # TODO 221114
        #     print("nan_318")
        #     # outputs_ = outputs # TODO
        #     # # outputs_[outputs_ != outputs_] = float("Inf")
        #     # outputs_[outputs_ != outputs_] = float(0)
        #     # outputs = outputs_
        #     outputs = torch.rand_like(outputs) #.requires_grad_()
        metrics = self.calculate_metrics(inputs, outputs)
        metrics['loss'] = self.calculate_loss(inputs, outputs).item() # inputs = 2 by 810 by 20 by 896 , outputs = 840 by 2

        # metric_name_for_writer_val = list(metrics.keys())
        # for name_i in metric_name_for_writer_val:
        #     self.writer.add_scalar(name_i + "_val", metrics[name_i], self.epoch)
        # self.writer.add_scalar("loss_val", metrics['loss'], self.epoch) # TODO 220929

        return metrics

    def predict(self, dataset, **loader_kwargs): # 마지막 예측해서 results 쓸때
        """
        Determine the outputs for all loaded data from the dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        inputs : Tensor
                 The exact inputs used to calculate the outputs (in case they were stochastic and need saving)
        outputs : Tensor
                  The outputs from each run of :function:`forward`
        """
        self.train(False)
        loader_kwargs.setdefault('batch_size', 1)
        dataset = self._make_dataloader(dataset, **loader_kwargs)

        pbar = tqdm.trange(len(dataset), desc="Predicting")
        data_iterator = iter(dataset)

        inputs = list()
        outputs = list()

        with torch.no_grad(): # TODO
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                output_batch = self.forward(*input_batch) #, epoch = self.epoch) # TODO 221018 221021

                inputs.append([tensor.cpu() for tensor in input_batch])
                if isinstance(output_batch, torch.Tensor):
                    outputs.append(output_batch.cpu())
                else:
                    outputs.append([tensor.cpu() for tensor in output_batch])

        def package_multiple_tensors(batches: list):
            if isinstance(batches[0], torch.Tensor):
                return torch.cat(batches)
            elif isinstance(batches[0], (tuple, list)):
                return [torch.cat(b) for b in zip(*batches)]

        return package_multiple_tensors(inputs), package_multiple_tensors(outputs)

    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            elif m == 'lr':
                start_message += " {}: {:.3e} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.3f} |".format(m, metrics[m])
        tqdm.tqdm.write(start_message)

    def save_best(self):
        """
        Create a snapshot of what is being currently trained for re-laoding with the :py:meth:`load_best()` method.

        Returns
        -------
        best : Any
               Whatever format is needed for :py:meth:`load_best()`, will be the argument provided to it.
        """
        return [{k: v.cpu() for k, v in self.__dict__[m].state_dict().items()} for m in self._trainables]

    def load_best(self, best):
        """
        Load the parameters as saved by :py:meth:`save_best()`.

        Parameters
        ----------
        best: Any
        """
        for m, state_dict in zip(self._trainables, best):
            self.__dict__[m].load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def _retain_best(self, old_checkpoint, metrics_to_check: dict, retain_string: str): # retain : 새거인가..
        if retain_string is None:
            return old_checkpoint
        best_checkpoint = old_checkpoint

        def found_best():
            tqdm.tqdm.write("Best {}. Retaining checkpoint...".format(retain_string))
            self.best_metric = metrics_to_check[retain_string]
            return self.save_best()

        if retain_string not in metrics_to_check.keys():
            print("Here")
            tqdm.tqdm.write("No metric {} found in recorded metrics. Not saving best.")
        if self.best_metric is None:
            best_checkpoint = found_best()
        elif retain_string == 'loss' and metrics_to_check[retain_string] <= self.best_metric:
            best_checkpoint = found_best()

            # # for saving best # TODO 221118
            # for m, state_dict in zip(self._trainables, best_checkpoint):
            #     self.__dict__[m].load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
            # path = 'checkpoints_pretraining'
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # tqdm.tqdm.write("Saving...")
            #
            # encoder_best = ConvEncoderBENDR((len(To1020.EEG_20_div) + 1) * 17, encoder_h=args.hidden_size)
            # contextualizer_best = BENDRContextualizer(encoder.encoder_h,
            #                                           layer_drop=experiment.bending_college_args.layer_drop)
            # encoder_best.save('checkpoints_pretraining/encoder_best.pt')
            # contextualizer_best.save('checkpoints_pretraining/contextualizer_best.pt')

        elif retain_string != 'loss' and metrics_to_check[retain_string] >= self.best_metric:
            best_checkpoint = found_best()

            # # for saving best # TODO 221118
            # for m, state_dict in zip(self._trainables, best_checkpoint):
            #     self.__dict__[m].load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
            # path = 'checkpoints_pretraining'
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # tqdm.tqdm.write("Saving...")
            # from dn3_ext_mh import BENDRClassification
            # model_best = BENDRClassification(2, 896, 20)
            # model_best.load
            from layers_mh import ConvEncoderBENDR, BENDRContextualizer
            # encoder_best = ConvEncoderBENDR(20, encoder_h=512)
            # encoder_best.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
            #                                   freeze_encoder=args.freeze_encoder)
            # contextualizer_best = BENDRContextualizer(encoder.encoder_h,
            #                                           layer_drop=experiment.bending_college_args.layer_drop)
            # encoder_best.save('checkpoints_pretraining/encoder_best.pt')
            # contextualizer_best.save('checkpoints_pretraining/contextualizer_best.pt')

        return best_checkpoint

    @staticmethod
    def _dataloader_args(dataset, training=False, **loader_kwargs):
        # Only shuffle and drop last when training
        loader_kwargs.setdefault('shuffle', training)
        loader_kwargs.setdefault('drop_last', training)
        loader_kwargs.setdefault('num_workers', 4)  # TODO 221108

        return loader_kwargs

    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        """Any args that make more sense as a convenience function to be set"""
        if isinstance(dataset, DataLoader):
            return dataset

        return DataLoader(dataset, **self._dataloader_args(dataset, training, **loader_kwargs))

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None,
            resume_epoch=None, resume_iteration=None, log_callback=None, validation_callback=None,
            epoch_callback=None, batch_size=8, warmup_frac=0.2, retain_best='loss',
            validation_interval=None, train_log_interval=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : DN3ataset, DataLoader
        validation_dataset : DN3ataset, DataLoader
        epochs : int
                 Total number of epochs to fit
        resume_epoch : int
                      The starting epoch to train from. This will likely only be used to resume training at a certain
                      point.
        resume_iteration : int
                          Similar to start epoch but specified in batches. This can either be used alone, or in
                          conjunction with `start_epoch`. If used alone, the start epoch is the floor of
                          `start_iteration` divided by batches per epoch. In other words this specifies cumulative
                          batches if start_epoch is not specified, and relative to the current epoch otherwise.
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        log_callback : callable
                       Function to run after every log interval that has signature: fn(train_metrics) -> None
        validation_callback : callable
                        Function to run after every time the validation dataset is run through. This typically has the
                        result of this and the `epoch_callback` called at the end of the epoch, but this is also called
                        after `validation_interval` batches.
                        This callback has the signature: fn(validation_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        batch_size : int
                     The batch_size to be used for the training and validation datasets. This is ignored if they are
                     provided as `DataLoader`.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        validation_interval: int, None
                             The number of batches between checking the validation dataset
        train_log_interval: int, None
                      The number of batches between persistent logging of training metrics, if None (default) happens
                      at the end of every epoch.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----
        If the datasets above are provided as DN3atasets, automatic optimizations are performed to speed up loading.
        These include setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. Unless you are very comfortable with PyTorch, it's probably better
        to not provide your own DataLoader, and let this be done automatically.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataset = self._make_dataloader(training_dataset, training=True, **loader_kwargs)

        if resume_epoch is None:
            if resume_iteration is None or resume_iteration < len(training_dataset):
                resume_epoch = 1
            else:
                resume_epoch = resume_iteration // len(training_dataset)
        resume_iteration = 1 if resume_iteration is None else resume_iteration % len(training_dataset)

        _clear_scheduler_after = self.scheduler is None
        if _clear_scheduler_after:
            last_epoch_workaround = len(training_dataset) * (resume_epoch - 1) + resume_iteration
            last_epoch_workaround = -1 if last_epoch_workaround <= 1 else last_epoch_workaround
            self.set_scheduler(
                torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, epochs=epochs,
                                                    steps_per_epoch=len(training_dataset),
                                                    pct_start=warmup_frac,
                                                    last_epoch=last_epoch_workaround)

                # for CosineAnnealingLR
                # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_0=10, T_mult=1, eta_min=0.0001,
                #                                     last_epoch=last_epoch_workaround)
                # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

                # for CosineAnnealingLR
                # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= 10, last_epoch=last_epoch_workaround)
                # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
            )

        validation_log = list()
        train_log = list()
        self.best_metric = None
        best_model = self.save_best()

        train_log_interval = len(training_dataset) if train_log_interval is None else train_log_interval
        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, iterations):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    try:
                        metrics[m] = (metrics[m] * (iterations - 1) + new_metrics[m]) / iterations
                    except KeyError:
                        metrics[m] = new_metrics[m]

        def print_training_metrics(epoch, iteration=None):
            if iteration is not None:
                self.standard_logging(metrics, "Training: Epoch {} - Iteration {}".format(epoch, iteration))
            else:
                self.standard_logging(metrics, "Training: End of Epoch {}".format(epoch))

        def _validation(epoch, iteration=None):
            _metrics = self.evaluate(validation_dataset, **loader_kwargs)

            if iteration is not None:
                self.standard_logging(_metrics, "Validation: Epoch {} - Iteration {}".format(epoch, iteration))
            else:
                self.standard_logging(_metrics, "Validation: End of Epoch {}".format(epoch))
            _metrics['epoch'] = epoch
            validation_log.append(_metrics)
            if callable(validation_callback):
                validation_callback(_metrics)
            return _metrics

        epoch_bar = tqdm.trange(resume_epoch, epochs + 1, desc="Epoch", unit='epoch', initial=resume_epoch, total=epochs)
        for epoch in epoch_bar:
            self.epoch = epoch
            pbar = tqdm.trange(resume_iteration, len(training_dataset) + 1, desc="Iteration", unit='batches',
                               initial=resume_iteration, total=len(training_dataset))
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = self._get_batch(data_iterator) # 60 x 135 x 768 , targets: 60
                train_metrics = self.train_step(*inputs) #, epoch = epoch) # TODO 221021
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                if 'momentum' in self.optimizer.defaults:
                    train_metrics['momentum'] = self.optimizer.param_groups[0]['momentum']
                update_metrics(train_metrics, iteration+1)
                pbar.set_postfix(metrics)
                train_metrics['epoch'] = epoch
                train_metrics['iteration'] = iteration
                train_log.append(train_metrics)
                if callable(step_callback):
                    step_callback(train_metrics)

                if iteration % train_log_interval == 0 and epoch_bar.total == epoch:  # pbar.total != iteration: # TODO
                    print_training_metrics(epoch, iteration)
                    train_metrics['epoch'] = epoch
                    train_metrics['iteration'] = iteration
                    if callable(log_callback):
                        log_callback(metrics)
                    metrics = OrderedDict()

                if isinstance(validation_interval, int) and (iteration % validation_interval == 0)\
                        and validation_dataset is not None:
                    _m = _validation(epoch, iteration) # _m = metrics
                    best_model = self._retain_best(best_model, _m, retain_best) # old_checkpoint, metrics_to_check: dict, retain_string: str == 'loss'

            # Make epoch summary
            metrics = DataFrame(train_log)
            metrics = metrics[metrics['epoch'] == epoch]
            metrics = metrics.mean().to_dict()
            metrics.pop('iteration', None)
            print_training_metrics(epoch)

            # TODO 221008
            metric_name_for_writer = list(metrics.keys())
            for name_i in metric_name_for_writer:
                self.writer.add_scalar(name_i + "_train", metrics[name_i], self.epoch)

            if validation_dataset is not None:
                metrics = _validation(epoch)
                metric_name_for_writer = list(metrics.keys())
                for name_i in metric_name_for_writer:
                    self.writer.add_scalar(name_i + "_val", metrics[name_i],
                                           self.epoch)  # TODO 221008
                best_model = self._retain_best(best_model, metrics, retain_best) # old_checkpoint, metrics_to_check: dict, retain_string: str == 'loss'

            if callable(epoch_callback):
                epoch_callback(metrics)
            metrics = OrderedDict()
            # All future epochs should not start offset in iterations
            resume_iteration = 1

            if not self.scheduler_after_batch and self.scheduler is not None:
                tqdm.tqdm.write(f"Step {self.scheduler.get_last_lr()} {self.scheduler.last_epoch}")
                self.scheduler.step()

        if _clear_scheduler_after:
            self.set_scheduler(None)
        self.epoch = None

        if retain_best is not None and validation_dataset is not None:
            tqdm.tqdm.write("Loading best model...")
            self.load_best(best_model)

        return DataFrame(train_log), DataFrame(validation_log)



class StandardClassification(BaseProcess):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=None, metrics=None, learning_rate=0.01,
                 label_smoothing=None, writer = None, **kwargs): # TODO 220929
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super(StandardClassification, self).__init__(cuda=cuda, lr=learning_rate, classifier=classifier,
                                                     metrics=metrics, **kwargs)
        if label_smoothing is not None and isinstance(label_smoothing, float) and (0 < label_smoothing < 1):
            self.loss = LabelSmoothedCrossEntropyLoss(self.classifier.targets, smoothing=label_smoothing).\
                to(self.device) # n_classes, smoothing=0.0, dim=-1
        elif loss_fn is None:
            self.loss = torch.nn.CrossEntropyLoss().to(self.device) # TODO : 221115
            # self.loss = torch.nn.BCELoss().to(self.device)
            # self.loss = torch.nn.BCEWithLogitsLoss().to(self.device) # TODO 221006 loss 바꿈
        else:
            self.loss = loss_fn.to(self.device)
        self.best_metric = None

        if writer is not None : # TODO : 220929
            self.writer = writer

    @staticmethod
    def _simple_accuracy(inputs, outputs):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        # average over last dimensions
        while len(outputs.shape) >= 3:
            outputs = outputs.mean(dim=-1)
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def forward(self, *inputs) :#, epoch): # here forward # TODO 221018 221021
        from dn3_utils_mh import min_max_normalize
        from copy import deepcopy
        inputs_c = list(deepcopy(inputs[:-1]))[0]
        inputs_c[inputs_c != inputs_c] = float(0) # TODO input nan 없애기 220926 221022
        inputs_c = tuple([inputs_c])
        # inputs_c = min_max_normalize(*inputs_c)

        if isinstance(self.classifier, Classifier) and self.classifier.return_features:
            # prediction, _ = self.classifier(*inputs[:-1])
            prediction, _ = self.classifier(*inputs_c) #, epoch=epoch) # 60 by 20 by 768 # TODO 221018 221021
            # prediction = self.classifier(*inputs[:-1])
        else:
            # prediction = self.classifier(*inputs[:-1])
            # prediction = self.classifier(*inputs[:-1]) #, epoch=epoch) # TODO 221018 221021
            prediction = self.classifier(*inputs_c)


        return prediction # TODO : nan_to_num 필요 0으로? # torch.Size([60, 2]) 220922

    def calculate_loss(self, inputs, outputs):
        inputs = list(inputs) # TODO calculate_loss here

        def expand_for_strided_loss(factors):
            inputs[-1] = inputs[-1].unsqueeze(-1).expand(-1, *factors)

        check_me = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if len(check_me.shape) >= 3:
            expand_for_strided_loss(check_me.shape[2:])

        return super(StandardClassification, self).calculate_loss(inputs, outputs)

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None, epoch_callback=None,
            batch_size=8, warmup_frac=0.2, retain_best='loss', balance_method=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : DN3ataset, DataLoader
        validation_dataset : DN3ataset, DataLoader
        epochs : int
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        batch_size : int
                     The batch_size to be used for the training and validation datasets. This is ignored if they are
                     provided as `DataLoader`.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        balance_method : (None, str)
                         If and how to balance training samples when training. `None` (default) will simply randomly
                         sample all training samples equally. 'undersample' will sample each class N_min times
                         where N_min is equal to the number of examples in the minority class. 'oversample' will sample
                         each class N_max times, where N_max is the number of the majority class.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----
        -----
        If the datasets above are provided as DN3atasets, automatic optimizations are performed to speed up loading.
        These include setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. Unless you are very comfortable with PyTorch, it's probably better
        to not provide your own DataLoader, and let this be done automatically.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        return super(StandardClassification, self).fit(training_dataset, epochs=epochs, step_callback=step_callback,
                                                       epoch_callback=epoch_callback, batch_size=batch_size,
                                                       warmup_frac=warmup_frac, retain_best=retain_best,
                                                       validation_dataset=validation_dataset,
                                                       balance_method=balance_method,
                                                       **loader_kwargs)

    BALANCE_METHODS = ['undersample', 'oversample', 'ldam']
    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        if isinstance(dataset, DataLoader):
            return dataset

        try:
            loader_kwargs.pop('_d') # TODO
        except:
            pass
        loader_kwargs = self._dataloader_args(dataset, training=training, **loader_kwargs)

        if training and loader_kwargs.get('sampler', None) is None and loader_kwargs.get('balance_method', None) \
                is not None:
            method = loader_kwargs.pop('balance_method')
            assert method.lower() in self.BALANCE_METHODS
            if not hasattr(dataset, 'get_targets'):
                print("Failed to create dataloader with {} balancing. {} does not support `get_targets()`.".format(
                    method, dataset
                ))
            elif method.lower() != 'ldam':
                sampler = balanced_undersampling(dataset) if method.lower() == 'undersample' \
                    else balanced_oversampling(dataset)
                # Shuffle is implied by the balanced sampling
                # loader_kwargs['shuffle'] = None
                loader_kwargs['sampler'] = sampler
            else:
                self.loss = create_ldam_loss(dataset)

        if loader_kwargs.get('sampler', None) is not None:
            loader_kwargs['shuffle'] = None

        # Make sure balance method is not passed to DataLoader at this point.
        loader_kwargs.pop('balance_method', None)

        return DataLoader(dataset, **loader_kwargs)



def get_label_balance(dataset):
    """
    Given a dataset, return the proportion of each target class and the counts of each class type

    Parameters
    ----------
    dataset

    Returns
    -------
    sample_weights, counts
    """
    assert hasattr(dataset, 'get_targets')
    labels = dataset.get_targets()
    counts = np.bincount(labels) # 빈도수 체크
    if counts == 0:
        print("processes_mh 800 ")
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    class_freq = counts/counts.sum()
    if len(counts) < 10:
        tqdm.tqdm.write('Class frequency: {}'.format(' | '.join('{:.2f}'.format(c) for c in class_freq)))
    else:
        tqdm.tqdm.write("Class frequencies range from {:.2e} to {:.2e}".format(class_freq.min(), class_freq.max()))
    return sample_weights, counts



def balanced_undersampling(dataset, replacement=False):
    tqdm.tqdm.write("Undersampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=replacement)


def balanced_oversampling(dataset, replacement=True):
    tqdm.tqdm.write("Oversampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=replacement)


class LDAMLoss(torch.nn.Module):
    # September 2020 - Originally taken from: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    # October   2020 - Modified to support non-cuda devices and a switch to activate drw

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        self._cls_nums = cls_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def _determine_drw_weights(self, beta=0.9999):
        effective_num = 1.0 - np.power(beta, self._cls_nums) # 1 - beta^_cls_nums
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        return torch.from_numpy(per_cls_weights / np.sum(per_cls_weights) * len(self._cls_nums)).float()

    def drw(self, on=True, beta=0.9999):
        self.weight = self._determine_drw_weights(beta=beta) if on else None

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :].to(index.device), index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        w = self.weight.to(index.device) if self.weight is not None else None
        return torch.nn.functional.cross_entropy(self.s * output, target, weight=w)


def create_ldam_loss(training_dataset):
    sample_weights, counts = get_label_balance(training_dataset)
    return LDAMLoss(counts)
