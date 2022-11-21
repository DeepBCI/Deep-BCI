import copy
import torch
import numpy as np
from torch import nn
import random
import mne
import parse
import tqdm
import torch.nn.functional as F
from math import ceil
from pathlib import Path
from dn3_data_dataset_mh import DN3ataset
from model_mh import Classifier
from layers_mh import _BENDREncoder, ConvEncoderBENDR, _Hax, BENDRContextualizer, _make_span_from_seeds, _make_mask, Flatten, Permute
from processes_mh import StandardClassification, BaseProcess


# FIXME this is redundant with part of the contextualizer
class EncodingAugment(nn.Module): # represent position using an additive grouped conv layer w. a receptive field of 25 and 16 groups
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        # conv = nn.utils.weight_norm(conv, dim=2) # TODO 220927 delete
        import weight_norm_mh
        # conv = nn.utils.weight_norm(conv, dim=2) # g = magnitude, v = direction # TODO _g _v : for better training ; 필요 없을지도... 그냥 나눠보는거 ㄱ
        conv = weight_norm_mh.weight_norm_mh(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape # batch size, feature dim, seq. num

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)

# pooling.. ; Linear model
class LinearHeadBENDR(Classifier):

    @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, x):
        from dn3_utils_mh import min_max_normalize
        # x = min_max_normalize(x)  #  중복 일듯. 확인 ㄱ -> by min max 확인하기.
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return self.extended_classifier(x)

    def __init__(self, targets, samples, channels, encoder_h=512, projection_head=False,
                 enc_do=0.1, feat_do=0.4, pool_length=4, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05,
                 mask_c_span=0.1, classifier_layers=1): # TODO enc_do :drop , feat_do = 0.4 drop
        if classifier_layers < 1:
            self.pool_length = pool_length
            self.encoder_h = 3 * encoder_h
        else:
            self.pool_length = pool_length // classifier_layers
            self.encoder_h = encoder_h
        super().__init__(targets, samples, channels)

        self.encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, projection_head=projection_head, dropout=enc_do)
        encoded_samples = self.encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        # Important for short things like P300
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        self.enc_augment = EncodingAugment(encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span,
                                           mask_t_span=mask_t_span)
        tqdm.tqdm.write(self.encoder.description(None, samples) + " | {} pooled".format(pool_length))
        self.summarizer = nn.AdaptiveAvgPool1d(pool_length)

        classifier_layers = [self.encoder_h * self.pool_length for i in range(classifier_layers)] if \
            not isinstance(classifier_layers, (tuple, list)) else classifier_layers
        classifier_layers.insert(0, 3 * encoder_h * pool_length) # 맨앞에 3*256 -> 두번째 feature 수 위해.
        self.extended_classifier = nn.Sequential(Flatten())
        for i in range(1, len(classifier_layers)):
            self.extended_classifier.add_module("ext-classifier-{}".format(i), nn.Sequential(
                nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                nn.Dropout(feat_do),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_layers[i]),
            ))

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(not freeze)
        print("Loaded {}".format(encoder_file))

    def load_pretrained_modules(self, encoder_file, contextualizer_file, strict=False, freeze_encoder=True):
        self.load_encoder(encoder_file, strict=strict, freeze=freeze_encoder)
        self.enc_augment.init_from_contextualizer(contextualizer_file)

# poolilng x ; BENDR model
class BENDRClassification(Classifier):

    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        # from dn3_utils_mh import min_max_normalize
        # x = min_max_normalize(x)  # 이미 함. 중복 220926
        encoded = self.encoder(x[0]) #  here
        # [60 by 20 by 768] -> [60 by 512 by 8]
        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded) # 60 by 512 by 9
        # return self.projection_mlp(context[:, :, 0])
        # return nn.functional.adaptive_max_pool1d(context, output_size=1)
        return context[:, :, -1] # 60 by 512 # 왜 -1인지 확인 -> permute[1,2,0] -> 맨 뒤에 있는게 가장 첫번째 것.

    def __init__(self, targets, samples, channels, encoder_h=512, contextualizer_hidden=3076, projection_head=False,
                 new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0, keep_layers=None,
                 mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False): # TODO 원래 dropout=0. , layer_drop=0
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super().__init__(targets, samples, channels)

        encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head) # TODO channels = 20
        encoded_samples = encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                                  mask_c_span=mask_c_span, dropout=dropout,
                                                  mask_t_span=mask_t_span)

        self.encoder = nn.DataParallel(encoder) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer) if multi_gpu else contextualizer

        tqdm.tqdm.write(encoder.description(sequence_len=samples))

        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(encoder_h, encoder_h),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings # num_embeddings, embedding_dim, padding_idx(optional)

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=False,
                                freeze_contextualizer=False, freeze_position_conv=False,
                                freeze_mask_replacement=True, strict=False):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)
        self.contextualizer.mask_replacement.requires_grad = freeze_mask_replacement
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False




class OnlyCNN(Classifier):
    @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, *x) : #, epoch): # TODO 221021
        x = x[0].clone().detach() #.requires_grad_(True)
        # x = torch.tensor(x[0])
        # if x.shape[0] < 60:
        #     a = (60 - x.shape[0]) ls// 2
        #     b = 60 - x.shape[0] - a
        #     x = F.pad(x, (0,0,0,0,a,b), "constant", 0)
        if x.shape[2] > 896:
            x=x[:,:,896]
        elif x.shape[2] < 896:
            a = (896-x.shape[2])//2
            b = 896-x.shape[2]-a
            x = F.pad(x, (a,b), "constant", 0)

        x = x[..., None]
        batchbatch = x.shape[0]
        x = x.reshape(batchbatch, 20, 32, -1)
        x = self.pool(F.relu(self.conv1(x)))  # -> 60, 512, 30, 26 -> 60 by 512 by 15 by 13
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x))) # -> 60 by 128 by 2 by 1
        x = torch.flatten(x, 1)  # -> 60 by 256
        x = F.relu(F.linear(x, self.weight1))
        # x = F.relu(F.linear(x, self.weight2))
        # x = F.relu(self.conv_l(x.unsqueeze(2)).squeeze(2))
        # x = F.relu(self.fc1(x)) # -> 60 by 128
        x = F.relu(self.fc2(x))# -> 60 by 64
        x = self.fc3(x)# -> 60 by 2

        return x

    def __init__(self, targets, samples, channels, encoder_h=512, projection_head=False,
                 pool_length=4, classifier_layers=1):

        self.pool_length = pool_length // classifier_layers
        self.encoder_h = encoder_h
        super().__init__(targets, samples, channels)
        self.samples = samples
        self.conv1 = nn.Conv2d(20, 512, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(512, 128, 2)

        # self.conv3 = nn.Conv2d(256, 128, 2)
        # self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        a = int(samples//32)
        b = int(samples // 64)
        # 128*3*(((a-2)//2 - 1)//2 -1)//2
        self.weight1 = nn.Parameter(torch.randn(128, 128*7*(((a-2)//2 - 1)//2))) #128*3*((((a-2)//2 - 1)//2 -1)//2)))
        self.weight2 = nn.Parameter(torch.randn(128, 128 * 7 * (((b - 2) // 2 - 1) // 2)))
        # self.conv_l = nn.Conv1d(128*7*(((a-2)//2 - 1)//2), 60, 1)
        # self.conv1 = nn.Conv2d(20, self.encoder_h, 3) # in_channel, out_channel, kernel
        self.summarizer = nn.AdaptiveAvgPool1d(pool_length)


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
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels),
                   **modelargs)  # target sample channel



class BendingCollegeWav2Vec(BaseProcess):
    """
    A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
    """
    def __init__(self, encoder, context_fn, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                 permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                 l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                 num_negatives=100, writer = None, **kwargs): # TODO : 221031
        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if multi_gpu:
            encoder = nn.DataParallel(encoder)
            context_fn = nn.DataParallel(context_fn)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(encoder_grad_frac * ig for ig in in_grad))
        super(BendingCollegeWav2Vec, self).__init__(encoder=encoder, context_fn=context_fn,
                                                    loss_fn=nn.CrossEntropyLoss(), lr=learning_rate,
                                                    l2_weight_decay=l2_weight_decay,
                                                    metrics=dict(Accuracy=self._contrastive_accuracy,
                                                                 Mask_pct=self._mask_pct), **kwargs)
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(context_fn, 'start_token', None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives
        if writer is not None : # TODO : 221031
            self.writer = writer

    def description(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples, self.mask_span, self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span))
        return desc

    def _generate_negatives(self, z): # z = encoder의 output
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape # 8, 512, 23
        z_k = z.permute([0, 2, 1]).reshape(-1, feat) # (184, 512)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives)) #size = (10, 2*23 = 460)
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size): # 원래 batch size = 64 -> 내가 8...
                negative_inds[i] += i * full_len

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat) # (184, 512) -> (8, 23, 20, 512)
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives): # (unmasked_z, c, negatives)
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2) # token 넣은 거 -> transformer encoder -> outcome : 1: - token 빼는 거
        # 원래 c: (10, 512, 24) -> (10, 23, 1, 512)
        z = z.permute([0, 2, 1]).unsqueeze(-2) # unmasked : 원래 (10, 512, 23) -> (10, 23, 1, 512)

        # negatives = (10, 160, 20, 512), c = (10, 160, 1, 512), negative_in_target = (10, 160, 20), targets = (10, 160, 21, 512)
        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1) # (10, 23, 20) <- (10, 23, 20, 512)
        targets = torch.cat([c, negatives], dim=-2) # (10, 23, 21, 512)
        # c : (10, 23, 1, 512), negatives : (10, 23, 20, 512)
        logits = F.cosine_similarity(z, targets, dim=-1) / self.temp # (10, 160, 21) # now : (10, 23, 21) # 21: 20(negatives)+1(원래)
        # logits : (10, 23, 21) = (batch, convolved features_dim, negative+1)
        # z: (10, 23, 1, 512), targets : (10, 23, 21, 512)
        if negative_in_target.any(): #no
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1]) # (1600, 21) # now : (230, 21)

    def forward(self, *inputs):
        z = self.encoder(inputs[0])

        if self.permuted_encodings: #no
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone() # (10, 512, 23)

        batch_size, feat, samples = z.shape

        if self._training:
            mask = _make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
        else: # no
            mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool) #(10, 3)
            half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
            if samples <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                              self.mask_span)] = True
        # mask : (10, 2)
        c = self.context_fn(z, mask) # (10, 512, 24) : token 넣은 거..

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(z) # (10, 23, 20, 512), (10, 460 = 23*20)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives) # unmasked_z: (10, 512, 23), c: (10, 512, 24), negatives : (10, 23, 20, 512)
        return logits, z, mask

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def _contrastive_accuracy(inputs, outputs):
        logits = outputs[0] # (184, 21)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return StandardClassification._simple_accuracy([labels], logits)

    def calculate_loss(self, inputs, outputs): # outputs = logits, z, mask = (30, 21), (10, 512, 3), (10, 3)
        logits = outputs[0] # (230, 21) # cosine similarity (30, 21)

        # mh ver
        # loss_mh = -torch.log(torch.exp(logits[:, 0])/torch.sum(torch.exp(logits[:, 1:]), dim = 1)+1e-9).mean() # (230)
        loss_mh = -torch.log(torch.exp(logits[:, 0]) / (torch.sum(torch.exp(logits), dim=1) + 1e-9)).mean()  # (230) (296) #221118

        return loss_mh + self.beta * outputs[1].pow(2).mean() # TODO : 221111

        # labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long) # (30) # TODO : 221109 의문 가득
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        # return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean() # cross entropy with cosine similarity , 0
