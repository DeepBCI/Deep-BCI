%% Topoplot

clear all; close all; clc; 

startup_bbci_toolbox('DataDir', MyDataDir);

dd = 'C:\Users\HANSEUL\Desktop\DoYeun\2019\analysis\';
filelist = 'gh_no';

%% Preprocessing
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); 
% Load cnt, mrk, mnt variables to Matlab

filtBank = [8 40];  % band pass filtering
ival = [-200 5000];

% Select channel
subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];

[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);

% IIR Filter (Band-pass filtering)
cnt = proc_filtButter(cnt, 2, filtBank);

%% cnt to epoch    
epo = cntToEpo(cnt, mrk, ival);

% baseline correction
base = [-200 0];
epo = proc_baseline(epo, base);
ival2 = [0 5000]; 
epo = proc_selectIval(epo,ival2);

%% Select class

epo_all = proc_selectClasses(epo, 'red', 'blue');
% epo_all = proc_selectClasses(epo, 'red', 'green');
% epo_all = proc_selectClasses(epo, 'blue', 'green');
% epo_all = proc_selectClasses(epo, 'red', 'green', 'blue');
% epo_all = proc_selectClasses(epo, 'white','red', 'green', 'blue', 'yellow', 'cyan', 'magenta');

%% Feature extraction - cross-validation
% Bandpower
% fv = proc_bandPower(epo_all, filtBank);
proc= struct();   
proc.train= ['[fv]= proc_bandPower(fv, [8 40]);'];
proc.apply= ['[fv]= proc_bandPower(fv, [8 40]);'];

%%
% RLDAshrink / forest, 10 fold corss validation
[C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo_all, 'RLDAshrink', 'proc', proc, 'kfold', 10);


%% Result

% plot_scalpPattern(epo, mnt, ival);
%Input:
% ERP: struct of epoched EEG data.
% MNT: struct defining an electrode montage
% IVAL: time interval for which scalp topography is to be plotted.
% OPTS: struct or property/value list of optional fields/properties:

[cnt, mrk, mnt] = eegfile_loadMatlab('./pilot_data/pilot_dhlee_2');

band= [4 8];
ival = [0 45000];
[b,a]= butter(5, band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);

epo = cntToEpo(cnt_flt, mrk, ival);


spec= proc_spectrum(epo, [4 8]);

H = scalpPlot(mnt, sepc);


