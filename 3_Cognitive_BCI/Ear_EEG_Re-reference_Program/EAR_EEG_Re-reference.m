clear all; clc;

AutobsseogDir = fullfile('C:','Program Files','MATLAB','R2013b','toolbox','eeglab13_6_5b','plugins','eeglab_plugin_aar-master');
MyToolboxDir = fullfile('C:','Program Files','MATLAB','R2013b','toolbox','bbci_public-master');
WorkingDir = fullfile('D:','Project','2016&2017_MA(Ear-EEG)','2.BrainProduct','brainproduct');
EegMyDataDir = fullfile(WorkingDir,'rawdata');

cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',EegMyDataDir,'TmpDir','/tmp/','History', 0);
cd(WorkingDir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subdir_list = {'20170526_LHT','20170530_CSR','20170530_LYS','20170602_SJY','20170605_CGY','20170608_PJH','20170608_KEI','20170620_KJS','20170622_PSJ','20170626_CSI','20170628_PSB','20170704_HIG','20170706_LJW','20170706_OHG','20170707_YHY'};
basename_list = {'MA1','MA2','MA3','MA4','MA5'};
stimDef= {1, 2; 'MA','BL'}; % Marker를 모두 포함하게 변경해야 함 % 1, 2는 각 task의 trigger
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for vp = 1 : length(subdir_list)
    cntDir = fullfile(EegMyDataDir,subdir_list{vp});
    rawDir=fullfile(cntDir,'rawdata');
    cd(rawDir);
    % Data load
    load cnt1; load cnt2; load cnt3; load cnt4; load cnt5;
    load mrk1; load mrk2; load mrk3; load mrk4; load mrk5;
    %% Channel Select - Ear (6)
    % 귀-뇌파 신호 선택
    cnt1=proc_selectChannels(cnt1, {'L1','L2','L3','R1','R2','R3'});
    cnt2=proc_selectChannels(cnt2, {'L1','L2','L3','R1','R2','R3'});
    cnt3=proc_selectChannels(cnt3, {'L1','L2','L3','R1','R2','R3'});
    cnt4=proc_selectChannels(cnt4, {'L1','L2','L3','R1','R2','R3'});
    cnt5=proc_selectChannels(cnt5, {'L1','L2','L3','R1','R2','R3'});
    %% Re-reference (5)
    for sessionIdx=1:5
        eval(['cnt' num2str(sessionIdx) '.CAR = cnt' num2str(sessionIdx) '.x - repmat(mean(cnt' num2str(sessionIdx) '.x,2),1,6);']); % CAR: common average reference
        eval(['cnt' num2str(sessionIdx) '.BiIpsi=[];']);    eval(['cnt' num2str(sessionIdx) '.BiCon=[];']);
        eval(['cnt' num2str(sessionIdx) '.BiAll=[];']);     eval(['cnt' num2str(sessionIdx) '.Cont=[];']);   eval(['cnt' num2str(sessionIdx) '.Ip=[];']);
        for ch=1:2 % Bipola-Ipsilateral
            for chIdx=1:3
                if chIdx>ch
                    eval(['cnt' num2str(sessionIdx) '.BiIpsi(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-Ipsilateral
                    eval(['cnt' num2str(sessionIdx) '.BiAll(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-All 
                end
            end
        end
        for ch=4:5
            for chIdx=4:6
                if chIdx>ch
                    eval(['cnt' num2str(sessionIdx) '.BiIpsi(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-Ipsilateral
                    eval(['cnt' num2str(sessionIdx) '.BiAll(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-All 
                end
            end
        end
        for ch=1:3
            for chIdx=4:6
                eval(['cnt' num2str(sessionIdx) '.BiCon(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-Contralateral
                eval(['cnt' num2str(sessionIdx) '.BiAll(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - cnt' num2str(sessionIdx) '.x(:,chIdx);']); % Bipolar-All
            end
        end
        for ch=1:6
            if ch<=3
                eval(['cnt' num2str(sessionIdx) '.Cont(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - mean(cnt' num2str(sessionIdx) '.x(:,4:6),2);']); % Contralateral-mean (Left-mean of right side)
                eval(['cnt' num2str(sessionIdx) '.Ip(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - mean(cnt' num2str(sessionIdx) '.x(:,1:3),2);']); % Ipsilateral(Left-mean of left side)
            else
                eval(['cnt' num2str(sessionIdx) '.Cont(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - mean(cnt' num2str(sessionIdx) '.x(:,1:3),2);']); % Contralateral-mean (Right-mean of left side)
                eval(['cnt' num2str(sessionIdx) '.Ip(:,end+1) = cnt' num2str(sessionIdx) '.x(:,ch) - mean(cnt' num2str(sessionIdx) '.x(:,4:6),2);']); % Ipsilateral(Right-mean of right side)
            end
        end
    end
end