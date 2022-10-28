clear all
close all
clc



%% concat channel

load('D:\BTS_dataset\KJW_word1_eeg_to_mat')


for i = 1:length(mnt.clab)
    EEG_data(:,i) = eval(['ch',num2str(i)]);
end

EEG_data = double(EEG_data);

clearvars -except EEG_data mnt mrk nfo dat
