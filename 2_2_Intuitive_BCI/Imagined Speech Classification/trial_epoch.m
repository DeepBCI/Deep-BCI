clear all
close all
clc



%% concat channel

load('D:\BTS_dataset\sub1')

% ch = 127, without ref == 127, etc --> emg
for ch = 1:length(mnt.clab)
    EEG_data(:,ch) = eval(['ch',num2str(ch)]);
end

EEG_data = double(EEG_data);

clearvars -except EEG_data mnt mrk nfo dat

concat_EEG = [];
% day3 --> end point
for i_time = 1:length(mrk.pos)
    concat_EEG(:,:,i_time) = EEG_data(mrk.pos(1,i_time)-2500:mrk.pos(1,i_time),:); % time series x channel x trial, -500ms for baseline-correction
end
% bl: -500ms ~ -200ms

FeatVect_EEG = concat_EEG;
label = mrk.toe';

clearvars -except FeatVect_EEG mnt mrk nfo dat label
save(['sub1_raw_EEG_file_trial.mat'],'-v7.3');
