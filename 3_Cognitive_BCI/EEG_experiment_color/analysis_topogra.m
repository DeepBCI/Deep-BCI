clear all; close all; clc;

startup_bbci_toolbox();

%% Data load
dd = 'C:\Users\HANSEUL\Desktop\DoYeun\2019\analysis\';
filelist = 'gh_frequency';

%% Preprocessing
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); 

filtBank = [8 40];
ival = [-200 5000];

 subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];

[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);
cnt = proc_filtButter(cnt, 2, filtBank);
% [b,a]= butter(5, band/cnt.fs*2);
% cnt_flt= proc_filt(cnt, b, a);

%% cnt to epoch

epo = cntToEpo(cnt, mrk, ival);

base = [-200 0];
epo = proc_baseline(epo, base);  
ival2 = [0 5000]; 
epo = proc_selectIval(epo,ival2);
% spec= proc_spectrum(epo, [4 8]);

epo_all = proc_selectClasses(epo, 'red', 'blue');

proc= struct();   
proc.train= ['[fv]= proc_bandPower(fv, [8 40]);'];
proc.apply= ['[fv]= proc_bandPower(fv, [8 40]);'];


[C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo_all, 'RLDAshrink', 'proc', proc, 'kfold', 10);


plot_scalpPattern(epo, mnt, ival);
H = plot_scalpPattern(epo, mnt, ival);


% Result after cross validation = 1-error rate
Result = 1 - C_eeg;
Result_Std = loss_eeg_std;  

% Cross-validation result 
Result*100
Result_Std*100

% Confusion matrix result 
[M, test_label] = max(out_eeg.out); % test label
[M, true_label] = max(epo_all.y); clear M; 
n = size(epo_all.y, 1); 

matrix_result = zeros(n, n); 


for i = 1:size(test_label, 3)       
for j = 1:length(true_label)
matrix_result(test_label(1, j, i), true_label(j)) = matrix_result(test_label(1, j, i), true_label(j)) + 1;
end       
end    


matrix_result = (matrix_result / sum(matrix_result(:, 1)));
matrix_result = matrix_result * 100;
matrix_result = matrix_result'; % true: y√‡, predicted: x√‡


