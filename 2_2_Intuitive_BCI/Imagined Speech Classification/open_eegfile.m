clear all
close all
clc



%% open eeg files

%% bbci
opt= [];
hdr= eegfile_readBVheader(['D:\BTS_dataset\day3_word_1']);

[cnt, mrk_orig]= eegfile_loadBV(['D:\BTS_dataset\day3_word_1']);



% chan_idx = 1:127; %1:64;
chan_idx = [];
remove = 1;

% remove_occipital_channel = [75,76,77,82,83,84] % PO9, O9, OI1h, OI2h, O10, PO10

% 128 --> ground
for i = 1:127
    if (i ~= 75)&&(i ~= 76)&&(i ~= 77)&&(i ~= 82)&&(i ~= 83)&&(i ~= 84)
        chan_idx(1,remove) = i;
        remove = remove+1;
    end
end

cnt = proc_selectChannels(cnt,chan_idx);






cnt = proc_filtButter(cnt, 5, [0.5 125]);
cnt = proc_filtnotch(cnt, cnt.fs, 60);
cnt = proc_filtnotch(cnt, cnt.fs, 120);



%% naming the markers
tmp = [];
what_tmp = [];
what_temp = [];
for i =1:9
    tmp{i,1} = ['S  ', num2str(i)];
end
for i =10:99
    tmp{i,1} = ['S ', num2str(i)];
end
for i =100:400
    tmp{i,1} = ['S', num2str(i)];
end




for ii = 1: 1:length(mrk_orig.desc)
    for iii = 1:400
        logical_value = strcmp(mrk_orig.desc{1,ii}, tmp{iii,1});
        if logical_value == 1
            what_temp(ii,1) = iii;
        end
    end
end

task_tmp = sort(what_temp);

what_task = countlabels(task_tmp);
numoftask = categories(what_task.Label);
numoftask = str2double(numoftask);
numoftask = sort(numoftask);
% numoftask = numoftask(2:end,1);

for i = 1:length(numoftask)
    stimDef{1,i} = numoftask(i,1);
    stimDef{2,i} = ['task',num2str(numoftask(i,1))];
end

mrk= mrk_defineClasses(mrk_orig, stimDef);

mrk.orig= mrk_orig;

mnt = getElectrodePositions(cnt.clab);

fs_orig= mrk_orig.fs;

var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr};


cnt.title= ['D:\BTS_dataset\day3_word1_eeg_to_mat'];

eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
    'channelwise',1, ...
    'format','double', ...
    'resolution', NaN);

disp('All EEG Data Converting is Done!, .eeg to mat');

