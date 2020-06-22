%% Initializing
clear all; close all; clc;

%% Data load
startup_bbci_toolbox();
dd = 'C:\Users\HANSEUL\Desktop\Analysis\Color\Converting\';
filelist = 'gh_no';

%% Preprocessing cnt
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); % Load cnt, mrk, mnt variables to Matlab

% Parameter setting\
rangeFreq = [8 40];
tinterval = 5000;
filtBank = rangeFreq;
% filtBank = [8 40];  % Setting frequency_band pass filtering
% ival = [-200 5000]; % sampling rate이 1000 이므로 마커 기준 2초를 잘라야 하니까 0~2000
ival = [-200 tinterval]; % sampling rate이 1000 이므로 마커 기준 2초를 잘라야 하니까 0~2000

[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);

% IIR Filter (Band-pass filtering)
cnt = proc_filtButter(cnt, 2, filtBank);

%% cnt to epoch
epo = cntToEpo(cnt, mrk, ival);

base = [-200 0];
epo = proc_baseline(epo, base);

% ival2 = [0 5000];
ival2 = [0 tinterval];
epo = proc_selectIval(epo,ival2);

%% Select class
% epo_all = proc_selectClasses(epo, 'red', 'blue');
epo_all = proc_selectClasses(epo, 'red', 'blue','green');
% epo_all = proc_selectClasses(epo, 'red', 'green');
% epo_all = proc_selectClasses(epo, 'blue', 'green');
% epo_all = proc_selectClasses(epo, 'white','red', 'green', 'blue', 'yellow', 'cyan', 'magenta');



%% Feature extraction - cross-validation
% basic multi-CSP
[csp_fv, csp_w, csp_eig] = proc_multicsp(epo_all, 3);
% [csp_fv, csp_w, csp_eig] = proc_csp_regularised(epo_all, 4, 1); % regulized CSP
% [csp_fv, csp_w] = proc_cspscp(epo_all, 2, 1); %CSP slow cortical potential variations
% [csp_fv, csp_w, csp_eig, t_filter] = proc_csssp(epo_all, 2); % Common Sparse Spectrum Spatial Pattern
% [csp_fv, csp_w] = proc_cspp_auto(epo_all); %auto csp patches, only for binary-class


proc= struct('memo', 'csp_w');  
proc.train= ['[fv, csp_w]= proc_multicsp(fv,3); ' ...
    'fv= proc_variance(fv); ' ...
    'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
    'fv= proc_variance(fv); ' ...
    'fv= proc_logarithm(fv);'];

% Bandpower
fv = proc_bandPower(epo_all, filtBank);

proc= struct();
proc.train = [strcat('[fv] = proc_bandPower(fv,[',string(rangeFreq(1)), ', ', string(rangeFreq(2)), ']);')];
proc.apply = [strcat('[fv]= proc_bandPower(fv,[', string(rangeFreq(1)), ', ', string(rangeFreq(2)), ']);')];
% proc.train= ['[fv]= proc_bandPower(fv, [8 40]);'];
% proc.apply= ['[fv]= proc_bandPower(fv, [8 40]);'];

%% RLDAshrink / forest, 10 fold corss validation
[C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo_all, 'RLDAshrink', 'proc', proc, 'kfold', 10);



%% Result

figure('Name', 'CSP Patterns');
plotCSPatterns(fv, mnt, csp_w, fv.y);

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
matrix_result = matrix_result'; % true: y축, predicted: x축

