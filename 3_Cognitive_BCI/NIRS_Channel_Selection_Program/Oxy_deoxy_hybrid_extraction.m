% single or multi

clear all; clc; close all;
% MyToolboxDir = fullfile('Z:','Project','2017','bbci_public-master');
startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
HwangDir = fullfile('D:','Project','2018','NIRS','Rawdata_converting');
NirsMyDataDir = fullfile(HwangDir,'rawdata','NIRS');

% cd(MyToolboxDir);
cd(HwangDir);

%% initial parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
% subdir_list = {'VP000','VP001','VP002','VP003','VP004','VP005','VP006','VP007','VP008','VP009','VP010','VP011','VP012','VP013','VP014','VP015','VP016','VP017','VP018','VP019','VP020'};
basename_list = {'MA1','MA2','MA3'};
class = {'MA','BL'};
crossval = [1 10]; % crossval(1) x crossval(2)-fold crossvalidation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for chlen = 1 % for combination only
    distance = proc_selectChannelLength(chlen); % 채널 길이 [1.5 / 2.12 / 3 / 3.35]
    for vp = 1 : length(subdir_list)
        disp([subdir_list{vp}, ' was started']);
        vpDir = fullfile(NirsMyDataDir,subdir_list{vp});
        
        %% Load NIRS data
        DistDir = fullfile(vpDir,['chlen',num2str(distance)]);
        cd(DistDir)
        load cntHb1; load cntHb2; load cntHb3;
        load mrk1; load mrk2; load mrk3;
        
        cd(HwangDir);
        [deoxy, mrk] = proc_appendCnt({cntHb1.deoxy, cntHb2.deoxy, cntHb3.deoxy},{mrk1, mrk2, mrk3});
        [oxy, ~]     = proc_appendCnt({cntHb1.oxy,   cntHb2.oxy,   cntHb3.oxy},  {mrk1, mrk2, mrk3});
        cntHb = proc_appendChannels(oxy, deoxy);
        clear cntHb1 cntHb2 cntHb3 mrk1 mrk2 mrk3 deoxy oxy;
        
        cd(HwangDir);
        
        %% BPF
        band_freq = [0.01 0.09];
        ord = 3;
        
        [z,p,k] = butter(ord, band_freq/cntHb.fs*2, 'bandpass');
        [SOS,G] = zp2sos(z,p,k);
        
        cntHb = proc_filtfilt(cntHb, SOS, G);
        
        %% Segmentation
        % marker selection
        [mrk, ev]= mrk_selectClasses(mrk, class);
        
        ival_epo  = [-11 35]*1000; % epoch range (unit: msec)
        epo = proc_segmentation(cntHb, mrk, ival_epo);
        
        %% baseline correction
        ival_base= [-5 -2]*1000; % for baseline correction
        epo = proc_baseline(epo, ival_base);
        
        %% feature vector
        
        ival_mean = [0 5; 5 10; 10 15]*1000;
        fv = proc_jumpingMeans(epo, ival_mean);
        fv.x = reshape(fv.x,[],60);
        save(['D:\Project\2022\NIRS_selection\Data\fv_' num2str(subdir_list{vp}) '_hybrid.mat'], 'fv');
    end
end