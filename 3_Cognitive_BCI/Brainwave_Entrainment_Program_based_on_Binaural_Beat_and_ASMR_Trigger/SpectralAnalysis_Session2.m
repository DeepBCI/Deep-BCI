%%
clear all; close all; clc;

%% data load
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
marker={'1','BB'; '2', 'ASMR'; '3', 'Comb'};


fs=250;
t_sti=180;   % stim duration
nTrial=3;

%% data path
file_path='D:\data';
file_name = '180920_dhlee_s2';
file=fullfile(file_path, file_name);

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file, {'device','brainVision';'marker',marker;'fs',fs});

cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

%%

% All channels
cnt=prep_selectChannels(cnt,{'Name',{'Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'}});
%,'Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'

% prefrontal
% cnt=prep_selectChannels(cnt,{'Name',{'Fp1','Fp2','F7','F8'}});
% frontal
% cnt=prep_selectChannels(cnt,{'Name',{'F3','Fz','F4'}});
% central
% cnt=prep_selectChannels(cnt,{'Name',{'C3','Cz','C4'}});
% temporal
% cnt=prep_selectChannels(cnt,{'Name',{'T7','T8','P7','P8'}});
% parietal
% cnt=prep_selectChannels(cnt,{'Name',{'P3','Pz','P4'}});
% occipital
% cnt=prep_selectChannels(cnt,{'Name',{'O1','O2'}});
%midline
% cnt=prep_selectChannels(cnt,{'Name', {'Fz','Cz','Pz'}});
% above each ear
% cnt=prep_selectChannels(cnt,{'Name',{'T7','T8'}});

% cnt=prep_selectChannels(cnt,{'Name', {'T7','P3'}});

cnt=prep_filter(cnt, {'frequency', [1 10]});
channel_layout=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19];
% channel_layout=[8 12 13 17];
cnt.chan=cnt.chan;
tmpC={};
cnt=proc_selectChannels(cnt,channel_layout);
for i =1:length(cnt.chan)
    if any(i==channel_layout)
        tmpC{end+1}=cnt.chan{i};
    end
end
cnt.chan=tmpC;
smt=prep_segmentation(cnt, {'interval', 1000*[1/fs t_sti]});

%% FFT_ASMR Comb2 only
figure (1)

% s  1: BB
subplot(3,1,1)
x=smt.x(:,find(smt.y_dec==1),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([0 20])
% ylim([0 0.00001])
title('BB')
xlabel('Frequency')
ylabel('Amplitude')
grid on;

% s  2: ASMR
subplot(3,1,2)
x=smt.x(:,find(smt.y_dec==2),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([0 20])
% ylim([0 polkp0.00001])
title('ASMR Trigger')
xlabel('Frequency')
ylabel('Amplitude')
grid on;

% s  3: Comb
subplot(3,1,3)
x=smt.x(:,find(smt.y_dec==3),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([0 20])
% ylim([0 0.00001])
title('Combined Stimulation')
xlabel('Frequency')
ylabel('Amplitude')
grid on;
%% Topo
figure(2)
visual_scalpPlot(smt, cnt,{'Ival' [[100 : 59900: 180000]]});

