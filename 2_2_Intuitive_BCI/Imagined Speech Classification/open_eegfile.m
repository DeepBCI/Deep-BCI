clear all
close all
clc



%% open eeg files


opt= [];
hdr= eegfile_readBVheader(['D:\BTS_dataset\day3_word_1']);

[cnt, mrk_orig]= eegfile_loadBV(['D:\BTS_dataset\day3_word_1']);



% chan_idx = 1:127; %1:64;
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

% chan_idx_check = 1:127;
% cnt_check = proc_selectChannels(cnt,chan_idx_check);




cnt = proc_filtButter(cnt, 5, [0.5 120]);
cnt = proc_filtnotch(cnt, cnt.fs, 60);


% 
% Fs = 1000;
% T = 1/Fs;
% L = length(cnt.x(:,10));
% t = (0:L-1)*T;
% 
% Y_cnt = fft(cnt.x(:,10));
% Y_P1 = abs(Y_cnt/L);
% Y_P2 = Y_P1(1:L/2+1,1);
% Y_P2(2:end-1) = 2*Y_P2(2:end-1);
% 
% Y_cntn = fft(cnt_notch.x);
% Yn_P1 = abs(Y_cntn/L);
% Yn_P2 = Yn_P1(1:L/2+1,1);
% Yn_P2(2:end-1) = 2*Yn_P2(2:end-1);
% 
% figure(2)
% subplot(2,1,1)
% plot(Y_P2(:,10))
% subplot(2,1,2)
% plot(Yn_P2(:,1))



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
for i =100:300
    tmp{i,1} = ['S', num2str(i)];
end




for i = 1: 1:length(mrk_orig.desc)
    for ii = 1:300
        logical_value = strcmp(mrk_orig.desc{1,i}, tmp{ii,1});
        if logical_value == 1
            what_temp(i,1) = ii;
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

% eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
%     'channelwise',1, ...
%     'format','int16', ...
%     'resolution', NaN);

cnt.title= ['D:\BTS_dataset\day3_word1_eeg_to_mat'];

eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
    'channelwise',1, ...
    'format','double', ...
    'resolution', NaN);

disp('All EEG Data Converting is Done!');

% concat ch
% for i = 1:length(mnt.clab)
%     EEG_data(:,i) = eval(['ch',num2str(i)]);
% end

% EEG_data = double(EEG_data);
