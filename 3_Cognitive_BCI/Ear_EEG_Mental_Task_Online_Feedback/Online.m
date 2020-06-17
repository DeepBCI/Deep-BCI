clear all; clc; close all;

MyToolboxDir = fullfile('C:','Program Files','MATLAB','R2013b','toolbox','bbci_public-master');
cd(MyToolboxDir);

RawDataDir = fullfile('D:\2019_Realtime\20170526_LHT');
startup_bbci_toolbox('DataDir',RawDataDir,'TmpDir','/tmp/');


OpenBMI % Edit the variable BMI if necessary
% startup_bbci_toolbox;

global BTB; % training시 필요
% BTB.DataDir = ['D:\2019_Realtime\20170526_LHT']; % training data 경로
% BMI.EEG_DIR = ['D:\BrainProduct\Test\20191202'];
% BMI.EEG_DATA = fullfile(BMI.EEG_DIR, '\Test');
subdir_list = {'20170526_LHT','20170530_CSR','20170530_LYS','20170602_SJY','20170605_CGY','20170608_PJH','20170608_KEI','20170620_KJS','20170622_PSJ','20170626_CSI','20170628_PSB','20170704_HIG','20170706_LJW','20170706_OHG','20170707_YHY'};
vp = 1;
%% func handle에 사용
% ----------- 오류발생 varargin 2cell에 0포함되서 오류
% if length(varargin) == 2
%     varargin(2) = [];
% end
%----------
%% Data Load
dataDir = fullfile('D:\2019_Realtime');
cntDir = fullfile(dataDir,subdir_list{vp});
cd(cntDir);
for idx = 1:5 % Number of sessions
    eval(['load cnt' num2str(idx)]);
    eval(['load mrk' num2str(idx)]);
end
% Merge
[cnt, mrk] = proc_appendCnt({cnt1, cnt2, cnt3, cnt4, cnt5}, {mrk1, mrk2, mrk3, mrk4, mrk5});
mnt = mnt_setElectrodePositions(cnt.clab);
mnt = mnt_setGrid(mnt, 'XXL');

cnt.y_logic=mrk.y;
%% Preprocessing
multi_csp = 1; repeat = 10; kfold = 10;

% filtering
switch multi_csp
    case 0
        disp('multiband csp가 적용되지 않았습니다.')
        [b, a] = butter(3, [8 13]/cnt.fs*2);
        cnt = proc_filtfilt(cnt, b, a);
    case 1
        disp('multiband csp가 적용되었습니다.')
        bands = [1 3; 4 7; 8 13; 14 29; 30 50];
        [b, a] = butters(3, bands/cnt.fs*2);
        cnt = proc_filterbank(cnt, b, a);
end

% epoching
ival_epo = [0 10]*1000; % ms (unit)
epo = proc_segmentation(cnt,mrk,ival_epo);
epo1 = proc_selectClasses(epo, {'MA','BL'});
%% Cross-validation module
online.train={
    'proc_multiBandSpatialFilter', {@proc_cspAuto,'patterns',2,'score', 'eigenvalues','selectPolicy','equalperclass'} %
    'proc_variance',{}
    'proc_logarithm',{}
    'func_train', {'classifier','LDA'} % @train_RLDAshrink (cross-validation), @sample_KFold, [repeat kfold]
    };
online.apply={
%     'prep_filter', {'frequency', [10 14]}
%     'func_projection',{}
%     'func_featureExtraction',{'feature', 'logvar'}
    'proc_multiBandSpatialFilter', {@proc_cspAuto,'patterns',2,'score', 'eigenvalues','selectPolicy','equalperclass'} %
    'proc_variance',{}
    'proc_logarithm',{}
    'classifier_applyClassifier',{}
    'func_predict', {'classifier','LDA'} % @train_RLDAshrink (cross-validation), @sample_KFold, [repeat kfold]
    
    };
online.option={
    'device', 'BrainVision'
    'paradigm', 'MotorImagery'
    'Feedback','on'
    'host', 'USER-PC' % 항상 체크할 것
    'port','51244'
    };

% cd('D:\2019_Realtime');
Classification_online(epo1, online);