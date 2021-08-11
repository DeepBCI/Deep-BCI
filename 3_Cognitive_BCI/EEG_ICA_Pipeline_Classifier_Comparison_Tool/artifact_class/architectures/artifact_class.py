import numpy as np
import torch.nn as nn
import torch.nn.functional as nnF

class CNN3L_legacy(nn.Module):
    def __init__(self, channel_size, epo_len, kernel_w, kernel_h, d2_out=16, d3_out=32, d4_out=64,
                 mp2_kernel_w=2, mp2_kernel_h=2, num_classes=2):
        super().__init__()

        def fcdimmer(inputd, kernel, stride=1, dilation=1, padding=0):
            return int((inputd+2*padding-dilation*(kernel-1)-1)/stride+1)

        def fcdimmer_recall(additional_call_time, inputd, kernel, stride=1, dilation=1, padding=0):
            x = fcdimmer(inputd, kernel ,stride, dilation, padding)
            for i in range(additional_call_time):
                x = fcdimmer(x, kernel, stride, dilation, padding)
            return x

        self.channel_size = channel_size
        self.epo_len = epo_len
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.pad_w = int((kernel_w - 1) / 2)
        self.pad_h = int((kernel_h - 1) / 2)

        d2_in = 1
        d5_in = d4_out * fcdimmer_recall(2, channel_size, mp2_kernel_w, mp2_kernel_w) * \
                fcdimmer_recall(2, epo_len, mp2_kernel_h, mp2_kernel_h)
        d5_out = d5_in
        # dim1 is channels, dim2 is time

        #self.dense1 = nn.Linear(channel_size*epo_len, d1_out)
        self.conv1 = nn.Conv2d(d2_in, d2_out, kernel_size=(kernel_w, kernel_h), padding=(self.pad_w, self.pad_h))
        self.bn1 = nn.BatchNorm2d(d2_out)
        self.conv2 = nn.Conv2d(d2_out, d3_out, kernel_size=(kernel_w, kernel_h), padding=(self.pad_w, self.pad_h))
        self.bn2 = nn.BatchNorm2d(d3_out)
        self.conv3 = nn.Conv2d(d3_out, d4_out, kernel_size=(kernel_w, kernel_h), padding=(self.pad_w, self.pad_h))
        self.bn3 = nn.BatchNorm2d(d4_out)
        #self.dense2 = nn.Linear(d5_in, d5_out)
        self.dense3 = nn.Linear(d5_out, num_classes)


    def forward(self, x):
        #x = x.view(x.size(0), -1)
        #x = F.relu(self.dense1(x))
        #x = x.view(x.size(0), -1, self.channel_size, self.epo_len)
        x = self.bn1(self.conv1(x))
        x = nnF.max_pool2d(nnF.relu(x), kernel_size=(2, 2))
        x = self.bn2(self.conv2(x))
        x = nnF.max_pool2d(nnF.relu(x), kernel_size=(2, 2))
        x = self.bn3(self.conv3(x))
        x = nnF.max_pool2d(nnF.relu(x), kernel_size=(2, 2))
        x = x.view(x.size(0), -1)  # Flatten
        #x = self.dense2(x)
        # x = F.dropout(x, training=self.training)
        #x = self.dense3(x)
        #x = F.sigmoid(self.dense3(x)) ## cross entropy applies softmax on its own, so it's not necessary
        x = self.dense3(x)

        return x

class CNN3LCustomParam(nn.Module):
    def __init__(self, convlayer_filtsizes: list, eeg_dim=None,
                 kernels=None, strides=None, dilations=None, padds=None, out_classes=2):
        super().__init__()

        convlayer_n = len(convlayer_filtsizes)
        # there should be same number of conv param sets as the number of conv layers
        if kernels is None:
            diff = convlayer_n - 1
            kernels = np.tile([[2,2]], (diff, 1))
        if strides is None:
            diff = convlayer_n - 1
            strides = np.tile([[1, 1]], (diff, 1))
        if dilations is None:
            diff = convlayer_n - 1
            dilations = np.tile([[1, 1]], (diff, 1))
        if padds is None:
            diff = convlayer_n - 1
            padds = np.tile([[0, 0]], (diff, 1))

        assert (convlayer_n == len(kernels))
        assert (convlayer_n == len(strides))
        assert (convlayer_n == len(dilations))
        assert (convlayer_n == len(padds))

        # calculate the I/O size for fully connected layers because, unfortunately,
        # they need to be predefined before actual network training.
        # ver2 : refit to accept arrays of kernels for a more customized design

        def fclayer_dimcalc(num_of_convlayers, input_dim, kernels=None, strides=None, dilations=None, padds=None):
            if kernels is None:
                kernels = [2,2]
            if strides is None:
                strides=[1,1]
            if dilations is None:
                dilations = [1,1]
            if padds is None:
                padds = [0,0]
            w, h = None, None
            for i in range(num_of_convlayers):
                # the 1st layer
                if i == 0:
                    w = np.floor((input_dim[0] + 2 * padds[i][0] - dilations[i][0] * (kernels[i][0] - 1) - 1)
                            / strides[i][0] + 1)
                    h = np.floor((input_dim[1] + 2 * padds[i][1] - dilations[i][1] * (kernels[i][1] - 1) - 1)
                            / strides[i][1] + 1)
                else:
                    w = np.floor((w + 2 * padds[i][0] - dilations[i][0] * (kernels[i][0] - 1) - 1) / strides[i][0] + 1)
                    h = np.floor((h + 2 * padds[i][1] - dilations[i][1] * (kernels[i][1] - 1) - 1) / strides[i][1] + 1)
                print('w is {0} h is {1}'.format(w,h))
            return int(w*h)

        c1_filtsize = 1 # initial layer has 1 (conv)channel size
        # fully connected input size is conv filter size of preceding layer X Width X Height
        # needs to be patched: maxpooling kernel values as argument
        kernels_calc = []
        strides_calc = []
        padding_calc = []
        dilations_calc = []
        for layeridx, layer in enumerate(convlayer_filtsizes):
            kernels_calc.append(kernels[layeridx])
            kernels_calc.append([2,2])
            strides_calc.append(strides[layeridx])
            strides_calc.append([2,2])
            padding_calc.append(padds[layeridx])
            padding_calc.append([0,0])
            dilations_calc.append(dilations[layeridx])
            dilations_calc.append([1,1])





        self.fc_inputsize =  convlayer_filtsizes[2] * \
                        fclayer_dimcalc(len(convlayer_filtsizes)*2, eeg_dim, kernels_calc, strides_calc, dilations_calc, padding_calc)

        # creating layers
        self.conv1 = nn.Conv2d(c1_filtsize, convlayer_filtsizes[0], kernel_size=(kernels[0][0], kernels[0][1]),
                               padding=(padds[0][0], padds[0][1]), stride=(strides[0][0], strides[0][1]), dilation=dilations[0])
        self.bn1 = nn.BatchNorm2d(convlayer_filtsizes[0])
        self.mp1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(convlayer_filtsizes[0], convlayer_filtsizes[1],
                               kernel_size=(kernels[1][0], kernels[1][1]), padding=(padds[1][0], padds[1][1]), dilation=dilations[1])
        self.bn2 = nn.BatchNorm2d(convlayer_filtsizes[1])
        self.mp2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(convlayer_filtsizes[1], convlayer_filtsizes[2],
                               kernel_size=(kernels[2][0], kernels[2][1]), padding=(padds[2][0], padds[2][1]), dilation=dilations[2])
        self.bn3 = nn.BatchNorm2d(convlayer_filtsizes[2])
        self.mp3 = nn.MaxPool2d((2, 2))
        self.dense3 = nn.Linear(self.fc_inputsize, out_classes)



    def forward(self, x):
        x = self.bn1(self.conv1(x))
        #print(x.shape)
        x = self.mp1(nnF.relu(x))
        #print(x.shape)
        x = self.bn2(self.conv2(x))
        #print(x.shape)
        x = self.mp2(nnF.relu(x))
        #print(x.shape)
        x = self.bn3(self.conv3(x))
        #print(x.shape)
        x = self.mp3(nnF.relu(x))
        #print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # x = self.dense2(x)
        # x = F.dropout(x, training=self.training)
        # x = self.dense3(x)
        # x = F.sigmoid(self.dense3(x)) ## cross entropy applies softmax on its own, so it's not necessary
        x = self.dense3(x)
        return x



class CNN3LCustomParam_v2(nn.Module):
    def __init__(self, convlayer_filtsizes: list, eeg_dim=None,
                 kernels=None, strides=None, dilations=None, padds=None, out_classes=2):
        super().__init__()

        convlayer_n = len(convlayer_filtsizes)
        # there should be same number of conv param sets as the number of conv layers
        if kernels is None:
            diff = convlayer_n - 1
            kernels = np.tile([[2,2]], (diff, 1))
        if strides is None:
            diff = convlayer_n - 1
            strides = np.tile([[1, 1]], (diff, 1))
        if dilations is None:
            diff = convlayer_n - 1
            dilations = np.tile([[1, 1]], (diff, 1))
        if padds is None:
            diff = convlayer_n - 1
            padds = np.tile([[0, 0]], (diff, 1))

        assert (convlayer_n == len(kernels))
        assert (convlayer_n == len(strides))
        assert (convlayer_n == len(dilations))
        assert (convlayer_n == len(padds))

        # calculate the I/O size for fully connected layers because, unfortunately,
        # they need to be predefined before actual network training.
        # ver2 : refit to accept arrays of kernels for a more customized design

        def fclayer_dimcalc(num_of_convlayers, input_dim, kernels=None, strides=None, dilations=None, padds=None):
            if kernels is None:
                kernels = [2,2]
            if strides is None:
                strides=[1,1]
            if dilations is None:
                dilations = [1,1]
            if padds is None:
                padds = [0,0]
            w, h = None, None
            for i in range(num_of_convlayers):
                # the 1st layer
                if i == 0:
                    w = int((input_dim[0] + 2 * padds[i][0] - dilations[i][0] * (kernels[i][0] - 1) - 1)
                            / strides[i][0] + 1)
                    h = int((input_dim[1] + 2 * padds[i][1] - dilations[i][1] * (kernels[i][1] - 1) - 1)
                            / strides[i][1] + 1)
                else:
                    w = int((w + 2 * padds[i][0] - dilations[i][0] * (kernels[i][0] - 1) - 1) / strides[i][0] + 1)
                    h = int((h + 2 * padds[i][1] - dilations[i][1] * (kernels[i][1] - 1) - 1) / strides[i][1] + 1)
                #print('w is {0} h is {1}'.format(w,h))
            return w*h

        c1_filtsize = 1 # initial layer has 1 (conv)channel size
        # fully connected input size is conv filter size of preceding layer X Width X Height
        # needs to be patched: maxpooling kernel values as argument
        kernels_calc = []
        strides_calc = []
        padding_calc = []
        dilations_calc = []
        for layeridx, layer in enumerate(convlayer_filtsizes):
            kernels_calc.append(kernels[layeridx])
            kernels_calc.append([2,2])
            strides_calc.append(strides[layeridx])
            strides_calc.append([2,2])
            padding_calc.append(padds[layeridx])
            padding_calc.append([0,0])
            dilations_calc.append(dilations[layeridx])
            dilations_calc.append([1,1])





        self.fc_inputsize =  convlayer_filtsizes[2] * \
                        fclayer_dimcalc(len(convlayer_filtsizes)*2, eeg_dim, kernels_calc, strides_calc, dilations_calc, padding_calc)

        # creating layers
        self.conv1 = nn.Conv2d(c1_filtsize, convlayer_filtsizes[0], kernel_size=(kernels[0][0], kernels[0][1]),
                               padding=(padds[0][0], padds[0][1]), stride=(strides[0][0], strides[0][1]), dilation=dilations[0])
        self.bn1 = nn.BatchNorm2d(convlayer_filtsizes[0])
        self.mp1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(convlayer_filtsizes[0], convlayer_filtsizes[1],
                               kernel_size=(kernels[1][0], kernels[1][1]), padding=(padds[1][0], padds[1][1]), dilation=dilations[1])
        self.bn2 = nn.BatchNorm2d(convlayer_filtsizes[1])
        self.mp2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(convlayer_filtsizes[1], convlayer_filtsizes[2],
                               kernel_size=(kernels[2][0], kernels[2][1]), padding=(padds[2][0], padds[2][1]), dilation=dilations[2])
        self.bn3 = nn.BatchNorm2d(convlayer_filtsizes[2])
        self.mp3 = nn.MaxPool2d((2, 2))
        self.dense3 = nn.Linear(self.fc_inputsize, out_classes)



    def forward(self, x):
        x = self.bn1(self.conv1(x))
        #print(x.shape)
        x = self.mp1(nnF.relu(x))
        #print(x.shape)
        x = self.bn2(self.conv2(x))
        #print(x.shape)
        x = self.mp2(nnF.relu(x))
        #print(x.shape)
        x = self.bn3(self.conv3(x))
        #print(x.shape)
        x = self.mp3(nnF.relu(x))
        #print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # x = self.dense2(x)
        # x = F.dropout(x, training=self.training)
        # x = self.dense3(x)
        # x = F.sigmoid(self.dense3(x)) ## cross entropy applies softmax on its own, so it's not necessary
        x = self.dense3(x)
        return x


