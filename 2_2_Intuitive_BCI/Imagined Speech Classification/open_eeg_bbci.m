clear all
close all
clc



%% open eeg files


opt= [];
hdr= eegfile_readBVheader(['D:\BTS_dataset\2022_10_11_BTS_denseNet_waveNet\KSW2_KJW8\2_word\sub8\day3\sub8_jwkim_day3_word_1']);

[cnt, mrk_orig]= eegfile_loadBV(['D:\BTS_dataset\2022_10_11_BTS_denseNet_waveNet\KSW2_KJW8\2_word\sub8\day3\sub8_jwkim_day3_word_1']);

chan_idx = [];
remove = 1;

% remove_occipital_channel = [75,76,77,82,83,84] % PO9, O9, OI1h, OI2h, O10, PO10

for i = 1:127
    if (i ~= 75)&&(i ~= 76)&&(i ~= 77)&&(i ~= 82)&&(i ~= 83)&&(i ~= 84)
        chan_idx(1,remove) = i;
        remove = remove+1;
    end
end

cnt = proc_selectChannels(cnt,chan_idx);