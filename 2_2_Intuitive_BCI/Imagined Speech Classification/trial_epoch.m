clear all
close all
clc



%% concat channel

load('D:\BTS_dataset\KJW_word1_eeg_to_mat')

% ch = 128, etc --> emg
for i = 1:length(mnt.clab)
    EEG_data(:,i) = eval(['ch',num2str(i)]);
end

EEG_data = double(EEG_data);

clearvars -except EEG_data mnt mrk nfo dat

concat_EEG = [];
% day3 --> end point
for i = 1:length(mrk.pos)
    concat_EEG(:,:,i) = EEG_data(mrk.pos(1,i)-1499:mrk.pos(1,i),:); % time series x channel x trial
end


FeatVect_EEG = concat_EEG;
label = mrk.toe';

clearvars -except FeatVect_EEG mnt mrk nfo dat label
save(['raw_EEG_file_trial.mat'],'-v7.3');
