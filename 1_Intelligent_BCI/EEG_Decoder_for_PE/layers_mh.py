import copy
import torch
import numpy as np
from torch import nn
import random
from dn3_utils_mh import min_max_normalize

import mne
import parse
import tqdm
import torch.nn.functional as F
from math import ceil
from pathlib import Path



# for control the random seed
random_seed = 2022
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


# layers editting...

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1) # contiguous == transpose랑 비슷한데 memory까지 순서 바꾸는 거 같음.


class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


def _make_span_from_seeds(seeds, span, total=None): # mask_seeds, span, total=total(==x.shape[-1]==seq_n)
    # pretrain : (samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int), self.mask_span
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span): # (0, 10), (18, 28)
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds) # 23개 중 10개씩


def _make_mask(shape, p, total, span, allow_no_inds=False): # (bs, seq), self.p_t, x.shape[-1]==n_seqs, self.mask_t_span
    # pretraining : (batch_size, samples), self.mask_rate, samples, self.mask_span = (8, 23), 0.065, 23, 10
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]): # batch_size
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0] # non-zero == True 인거 indices를 return [0] --> 23 길이 중에 p 확률로 index 뽑음
        # 위 == p의 확률로 mask_seeds <- indices 넣음
        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False): # unfreeze(F) ; default -> grad 추적 x, unfreeze(=T) -> grad 추적 o
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    # def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
    #              dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)): # TODO 원래 dropout = 0.
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2)): # TODO 221110
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width) # 뒤에가 False면 AssertionError

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        # in_features = 260 # TODO 221109 221115
        # in_features = 340  # TODO 221117 3 case - freq dir
        in_features = 20 # TODO 221117 : downstream case

        # self.conv1 = nn.Conv1d(in_features, encoder_h, 3, stride=3, padding=3 // 2)
        # self.conv2 = nn.Conv1d(encoder_h, encoder_h, 2, stride=2, padding=2 // 2)
        # for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
        #     if i == 0:
        #         self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
        #             self.conv1,
        #             nn.Dropout2d(dropout),
        #             nn.GroupNorm(encoder_h // 2, encoder_h),
        #             nn.GELU(),
        #         )) # layer_name, layer
        #     else:
        #         self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
        #         self.conv2,
        #         nn.Dropout2d(dropout),
        #         nn.GroupNorm(encoder_h // 2, encoder_h),
        #         nn.GELU(),
        #     ))  # layer_name, layer

        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            )) # layer_name, layer
            in_features = encoder_h


        if projection_head:
            print("projection_head = True...! in layers_mh.py")
            # self.encoder.add_module("projection-1", nn.Sequential(
            #     nn.Conv1d(in_features, in_features, 1),
            #     nn.Dropout2d(dropout*2),
            #     nn.GroupNorm(in_features // 2, in_features),
            #     nn.GELU()
            # ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = np.ceil(samples / factor)
        return samples

    def forward(self, x):
        # x = min_max_normalize(x) # TODO
        # print("BENDREncoder conv1") # 221115
        # print(self.conv1.weight.grad)
        # print("BENDREncoder conv2")
        # print(self.conv2.weight.grad)
        return self.encoder(x) # x : (60, 260, 15), output : (60, 512, 15) now: case3) (8, 512, 37)

        # enc_width = self._width
        # enc_downsample = self._downsampling
        # encoder_h = self.encoder_h
        # dropout = 0
        #
        # for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
        #     x = self.conv_layers[i](x)
        #     x = self.dropout_layers[i](x)
        #     x = self.groupnorm_layers[i](x) # 1.1439...?
        #     x = self.gelu_layers[i](x)
        #     # layer_name, layer
        # return x # 220926

class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed""" # to remove the need for layer normalization and warmup
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module): # Transformer 전처리 + transformer

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False): # TODO 원래 layer_drop=0.0
        super(BENDRContextualizer, self).__init__()

        self.dropout = dropout # layerdrop
        self.in_features = in_features
        self._transformer_dim = in_features * 3 # 1536

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation) # d_model = num of expected features in the input.
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim) # normalized_shape
        # layers = 4 # TODO 221108
        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)), #mean, std
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0 # position_encoder = 25 = receptive filed of 25 and 16 groups
        if position_encoder: # True

            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16) # in_channel, out_channel, kernel_size(=receptive field) =25
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim) # 1536 # set conv.weight = N(0, ~) TODO : stddev = 0.0001 , 2 / self._transformer_dim
            nn.init.constant_(conv.bias, 0) # set conv.bias = 0
            import weight_norm_mh
            # conv = nn.utils.weight_norm(conv, dim=2) # g = magnitude, v = direction # TODO _g _v : for better training ; 필요 없을지도... 그냥 나눠보는거 ㄱ
            conv = weight_norm_mh.weight_norm_mh(conv, dim=2)
            # with torch.no_grad():
            #     conv.weight.div_()

            # self.conv_layer = conv
            # self.gelu1 = nn.GELU()
            self.relative_position = nn.Sequential(conv, nn.GELU()) # nn.LayerNorm(in_features) # TODO conv.weight 확인 _g, _v
            # self.relative_position_norm_layer = nn.LayerNorm(in_features)

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]), # axis
            nn.LayerNorm(in_features),
            nn.Dropout(dropout), #0.15
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params) #e.g. net.apply(init_weights)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d): # 원래 주석처리 되어 있었음
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        # x = min_max_normalize(x)  # TODO
        bs, feat, seq = x.shape # batch_size, n_features, n_seqs || downstream : (60, 512, 1)
        if self.training and self.finetuning: # pretrain :no, downstream : yes
            if mask_t is None and self.p_t > 0: #yes
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span) # yes zero OR True #
            if mask_c is None and self.p_c > 0: #yes
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span) # yes zero OR True

        # Multi-gpu workaround, wastes memory
        x = x.clone() # (60, 512, 1)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            # y = x.clone()
            # x = self.conv_layer(x)
            # x = self.gelu1(x)
            # x = x + y  # TODO
            x = x + self.relative_position(x)

        x = self.input_conditioning(x) # (10, 512, 1) -> (1, 10, 1536) # transformer에 넣기 위해

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0) # 8, 60, 1536 -> 9, 60, 1536 # now : (1, 10, 1536) -> (2, 10, 1536)

        for layer in self.transformer_layers: # encoder 8 개짜리
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        # print("Contextualizer output_layer")
        # print(self.output_layer.weight.grad) #221115

        return self.output_layer(x.permute([1, 2, 0])) # pretrain : (10, 1536, 4) ->(10, 512, 4) || downstream: (60, 512, 1) -> (2, 60, 1536)

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)



