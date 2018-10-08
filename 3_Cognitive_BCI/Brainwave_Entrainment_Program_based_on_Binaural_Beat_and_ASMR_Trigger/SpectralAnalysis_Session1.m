%%
clear all; close all; clc;

%% data load
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
marker={'1','Comb 1'; '2', 'Comb 2'; '3', 'Comb 3'};


fs=250;
t_sti=180;   % stim duration
nTrial=3;

% freq = {[0.5 4], [4 8], [8 12], [12 15], [15 30]}; 
%% data path
file_path='H:\data';
file_name = '180920_jjkim_s1';
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

cnt=prep_filter(cnt, {'frequency', [2 10]});
channel_layout=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19];
% channel_layout= [5 8 10 12 15];
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

% s  1: Comb 1
subplot(3,1,1)
x=smt.x(:,find(smt.y_dec==1),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([2 12])
xticks([2:1:11])
ylim([0 0.000015])
title('Combined Stimulation 1')
xlabel('Frequency')
ylabel('Amplitude')
grid on;

% s  1: Comb 1
subplot(3,1,2)
x=smt.x(:,find(smt.y_dec==2),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([2 12])
xticks([2.2:1:11.2])
ylim([0 0.000015])
title('Combined Stimulation 2')
xlabel('Frequency')
ylabel('Amplitude')
grid on;

% s  3: Comb 3
subplot(3,1,3)
x=smt.x(:,find(smt.y_dec==3),:);
[X, f] = positiveFFT(x,fs);
Pxx=mean(mean(X,3),2);
plot(f,Pxx)
xlim([2 12])
xticks([2.2:1:11.2])
ylim([0 0.000015])
title('Combined Stimulation 3')
xlabel('Frequency')
ylabel('Amplitude')
grid on;
%% Topo
figure(2)
visual_scalpPlot(smt, cnt,{'Ival' [[100 : 179900: 180000]]});

