import copy
from copy import deepcopy
import mne
import parse
import parse
import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil
from pathlib import Path
from dn3_data_dataset_mh import DN3ataset

from layers_mh import *

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)


# for control the random seed
random_seed = 2022
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`DN3BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
    and *classification*.
    """
    def __init__(self, samples, channels, return_features=True):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):

        return None

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        print("Creating {} using: {} channels with trials of {} samples at {}Hz".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)

# models.py editting
class Classifier(DN3BaseModel):
    """
    A generic Classifer container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):

        if hasattr(dataset, 'get_targets'):
            targets = len(np.unique(dataset.get_targets()))
        elif dataset.info is not None and isinstance(dataset.info.targets, int):
            targets = dataset.info.targets
        else:
            targets = 2
        modelargs.setdefault('targets', targets)
        print("Creating {} using: {} channels x {} samples at {}Hz | {} targets".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq,
                                                                                        modelargs['targets']))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs) # target sample channel
        # downstream : spe : 896, 20,

    def __init__(self, targets, samples, channels, return_features=True):
        super(Classifier, self).__init__(samples, channels, return_features=return_features) # target sample channel
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        # x = min_max_normalize(x)
        features = self.features_forward(*x) # x 정상, features 이상 # here
        # return features
        if self.return_features:  # TODO : 221023
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def make_new_classification_layer(self):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.

        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.

        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the
        target should override this method, and there should be a variable called `self.classifier`

        """
        self.flat = Flatten()
        self.classifier_pre = nn.Linear(self.num_features_for_classification, 512) # 512, 2, bias=True # TODO 221121
        nn.init.xavier_normal_(self.classifier_pre.weight)
        # self.classifier_pre.bias.data.zero_()

        self.classifier = nn.Linear(512, self.targets)  # 512, 2, bias=True # TODO 221121
        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

        # sig_layer = nn.Sigmoid()
        # self.classifier = nn.Sequential(Flatten(), classifier_pre, classifier) #, nn.Sigmoid()) # : self.classifier 정의# TODO 221115


        # classifier = nn.Linear(self.num_features_for_classification, self.targets)  # 2048, 2, bias=True # TODO 221121
        # nn.init.xavier_normal_(classifier.weight)
        # classifier.bias.data.zero_()
        #
        # # sig_layer = nn.Sigmoid()
        # self.classifier = nn.Sequential(Flatten(), classifier)  # , nn.Sigmoid()) # : self.classifier 정의# TODO 221115


        # # 221023 input = 60 by 512 ====> 60, 32, 16
        # self.conv1 = nn.Conv2d(32, 16, 2, padding=3)
        # self.conv2 = nn.Conv2d(16, 8, 2, padding=3)
        # self.conv3 = nn.Conv2d(8, 4, 2, padding=3)
        # self.pool = nn.MaxPool2d(2,2)
        # # self.fc1 = nn.Linear(64, 32, bias = True)
        # # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # # self.fc2 = nn.Linear(32, 2, bias=True)
        # # torch.nn.init.xavier_uniform_(self.fc2.weight)
        #
        # self.fc_last = nn.Linear(64, 2, bias = True)
        # torch.nn.init.xavier_uniform_(self.fc_last.weight)

        # classifier -> Transformer decoder로 변형
        # decoder_layer = nn.TransformerDecoderLayer(self.num_features_for_classification, 8, activation = 'gelu',) # d_model, nhead = 8


    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        super(Classifier, self).freeze_features(unfreeze=unfreeze)

        if isinstance(self.classifier, nn.Module) and not freeze_classifier: # freeze 안함
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def classifier_forward(self, features):
        x = self.flat(features)
        x = self.classifier_pre(x)
        x = F.relu(x)
        return self.classifier(x)

        # return self.classifier(features) # downstream : (60, 512) -> (60, 2)
        # # features : 60, 512 # TODO 221023
        # batch_s = features.shape[0]
        # features = features.unsqueeze(2).unsqueeze(3).reshape(batch_s, 32, 4, 4) # ===> 60 by 2 되야 함
        # features = self.pool(F.relu(self.conv1(features)))  # -> 60, 16, 4, 4
        # features = self.pool(F.relu(self.conv2(features))) # -> 60, 4, 4, 4
        # features = self.pool(F.relu(self.conv3(features))) # 60,4,4,4
        # features = torch.flatten(features, 1) # 60, 64
        # features = self.fc_last(features)
        # return features
        # features = F.relu(self.fc1(features))  # -> 60 by 32
        # features = F.relu(self.fc2(features))  # -> 60 by 2
        # features = self.fc3(features)


    def features_forward(self, x):
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)


class BENDRClassifier(Classifier): # nn.Module

    def __init__(self, targets, samples, channels,
                 return_features=True,
                 encoder_h=256, # original sampling rate
                 encoder_w=(3, 2, 2), #, 2, 2, 2),
                 encoder_do=0., # drop # TODO encoder_do=0.
                 projection_head=False,
                 encoder_stride=(3, 2, 2), #, 2, 2, 2),
                 hidden_feedforward=3076,
                 heads=8,
                 context_layers=8,
                 context_do=0.15,
                 activation='gelu',
                 position_encoder=25,
                 layer_drop=0.,# TODO 원래 layer_drop=0.
                 mask_p_t=0.1,
                 mask_p_c=0.004,
                 mask_t_span=6,
                 mask_c_span=64,
                 start_token=-5,
                 **kwargs):
        self._context_features = encoder_h
        super(BENDRClassifier, self).__init__(targets, samples, channels, return_features=return_features)
        self.encoder = ConvEncoderBENDR(in_features=channels, encoder_h=encoder_h, enc_width=encoder_w,
                                        dropout=encoder_do, projection_head=projection_head,
                                        enc_downsample=encoder_stride)
        self.contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=hidden_feedforward, heads=heads, # Transformer
                                                  layers=context_layers, dropout=context_do, activation=activation,
                                                  position_encoder=position_encoder, layer_drop=layer_drop,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, mask_t_span=mask_t_span,
                                                  finetuning=True)


    @property
    def num_features_for_classification(self):
        return self._context_features

    def easy_parallel(self):
        if torch.cuda.device_count() > 1: # editted
            self.encoder = nn.DataParallel(self.encoder)
            self.contextualizer = nn.DataParallel(self.contextualizer)
            self.classifier = nn.DataParallel(self.classifier)

    def features_forward(self, x):
        # x = min_max_normalize(x)  #
        x = self.encoder(x)
        x = self.contextualizer(x)
        return x[0]
