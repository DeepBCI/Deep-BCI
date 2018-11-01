clc; clear; close all;
scripts_dir = '.\samples_comparison';
scripts_dir = '.\kh_subsets';
subject = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07',...
    'sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14',...
    'sub-15','sub-16'};

%%
% stuff: pre, post, pre-surrogates, and post-surrogates

phase_freq = 7:13;
amp_freq = 34:2:100;
surr=0;
if surr ==1
    PAC_post = 'matrix_post_PLV_surrogates.mat';
    PAC_pre = 'matrix_pre_PLV_surrogates.mat';
else
    PAC_post = 'matrix_post_tort.mat';
    PAC_pre = 'matrix_pre_tort.mat';
end

grandavgA = [];
for nsb = 1:length(subject)
    
    fname = [scripts_dir '\' subject{nsb} '\' PAC_post];
    load(fname);
    if surr == 1
        MI_post_comap = kh_matrix_post_surr;
    else
        MI_post_comap = kh_matrix_post;
    end
    grandavgA = cat(3, grandavgA, MI_post_comap);
end
disp('Collecting MI-post PLV done.');

grandavgB = [];
for nsb=1:length(subject)
    fname = [scripts_dir '\' subject{nsb} '\' PAC_pre];
    load(fname);
    if surr == 1
       MI_pre_comap = kh_matrix_pre_surr; 
    else
       MI_pre_comap = kh_matrix_pre;
    end
    grandavgB = cat(3, grandavgB, MI_pre_comap);
end
disp('Collecting MI-pre tort done.');

dat = [];
dat.val = permute(grandavgA, [3 1 2]);
dat.val2 = permute(grandavgB, [3 1 2]);

[corrected, critical] = kh_multiple_comparison(dat, 0.05, 'cluster-tval', 'same');