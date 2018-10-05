clear all;
% close all;
clc;

% sublist = {'Nskwak_2013_06_24','Hjpark_2013_06_25' , 'Shpark_2013_06_28' , 'Mhlee_2013_07_01', 'Bjkim_2013_07_02' ,...
%           'Jwkim_2013_07_02','Jhkim_2013_07_03','Jswoo_2013_07_04','Ihkim_2013_07_05','Jyeom_2013_07_06'};

% sublist = {'Shpark_2013_06_28' , 'Mhlee_2013_07_01', 'Bjkim_2013_07_02' ,...
%     'Jwkim_2013_07_02','Jhkim_2013_07_03','Jswoo_2013_07_04','Ihkim_2013_07_05',...
%     'Jyeom_2013_07_06','Yssin_2013_07_07','Cshwang_2013_07_08','Dmkim_2013_07_09'};

sublist = {'Shpark_2013_06_28' , 'Mhlee_2013_07_01','Jwkim_2013_07_02','Jhkim_2013_07_03','Yssin_2013_07_07',...
    'Cshwang_2013_07_08','Dmkim_2013_07_09','Jpkim_2013_07_10','Jhchoi_2013_07_10','Sggang_2013_07_10',...
    'Hjpark_2013_07_11','Mscho_2013_07_11','Mbkwon_2013_07_12','Dowon_2013_07_18'};

% sublist = {'Jswoo_2013_07_04','Ihkim_2013_07_05','Jyeom_2013_07_06','Yssin_2013_07_07','Cshwang_2013_07_08'};

for s=5:7
    [cnt,mrk,mnt] = eegfile_loadMatlab(['E:\EEG Data\Journal of Neural ENG\Converted EEG\' sublist{s}]);
    % load(sublist{s});
    % cnt = proc_selectChannels(cnt,{'P3','Pz','P4','CP1','CP2','PO9','O1','EMG','Throttle','Brakevalve'});
    % cnt = proc_selectChannels(cnt,'not',{'AF3','AF4'});
    % selectedCh = channelSelection(cnt);
    % cnt = proc_selectChannels(cnt,selectedCh);
    % cnt =  proc_commonAverageReference(cnt);
    cnt.x(:, end) = abs((cnt.x(:, end) + 1) / 2);
    y = medfilt1([zeros(mrk.fs*2.5, 1); cnt.x(:, end)], mrk.fs*5+1, 10000);
    cnt.x(:, end) = max(0, cnt.x(:, end)-y(1:(size(cnt.x, 1)))) + 0.0001*randn(size(cnt.x, 1), 1);
    cnt.x(:, end-1) = abs((cnt.x(:, end-1) + 1) / 2);
    y = medfilt1([zeros(mrk.fs*2.5, 1); cnt.x(:, end-1)], mrk.fs*5+1, 10000);
    cnt.x(:, end-1) = max(0, cnt.x(:, end-1)-y(1:(size(cnt.x, 1)))) + 0.0001*randn(size(cnt.x, 1), 1);
    
    for j=1:size(cnt.clab,2)
        if strcmp(cnt.clab(1,j),'EMG')
            ind = j;
        end
    end
    cnt_ = cnt;
    cnt_.x = cnt.x(:,1:ind-1);
    cnt_1 = cnt;
    cnt_1.x = cnt.x(:,ind);
    cnt_flt_em = cnt_1;
    cnt_flt = cnt_;
    % [filt_b,filt_a]= butter(5, 10/cnt.fs*2,'low');
    % cnt_flt_em= proc_filt(cnt_1, filt_b, filt_a);
    % cnt_flt_em = proc_rectifyChannels(cnt_flt_em);
    % [filt_b, filt_a]= butter(5, 1/cnt.fs*2,'high');
    % [filt_b, filt_a]= butter(5,[5/cnt.fs*2 40/cnt.fs*2],'stop');
    % cnt_flt= proc_filt(cnt_, filt_b, filt_a);
    % cnt_flt = proc_envelope(cnt_, 'ms_msec', 250);
    % cnt_flt = proc_movingAverage(cnt_, 100, 'centered');
    cnt_flt.x = cat(2,cnt_flt.x,cnt_flt_em.x,cnt.x(:,ind+1:end));
    
    mnt_70 = NaN;
    mnt_71 = NaN;
    mnt.x = cat(1,mnt.x,mnt_70,mnt_71);
    mnt.y = cat(1,mnt.y,mnt_70,mnt_71);
    mnt_70 = [NaN;NaN;NaN];
    mnt_71 = [NaN;NaN;NaN];
    mnt.pos_3d = cat(2,mnt.pos_3d,mnt_70,mnt_71);
    mnt_70 = 'Throttle';
    mnt_71 = 'Brakevalve';
    mnt.clab = cat(2,mnt.clab,mnt_70,mnt_71);
    mnt_70 = [NaN;NaN];
    mnt_71 = [NaN;NaN];
    mnt1 = mnt.box(1,end);
    mnt2 = mnt.box(2,end);
    mnt.box = mnt.box(:,1:size(mnt.box,2)-1);
    mnt.box = cat(2,mnt.box,mnt_70,mnt_71);
    l = size(mnt.box,2)+1;
    mnt.box(1,l) = mnt1;
    mnt.box(2,l) = mnt2;
    mnt_70 = [1;1];
    mnt_71 = [1;1];
    mnt.box_sz = cat(2,mnt.box_sz,mnt_70,mnt_71);
    
    %% Stimlus list
    
    mrk_train = mrk_selectEvents(mrk, 1:floor(length(mrk.pos)/2));
    %     mrk_train = mrk_selectEvents(mrk, 1:floor(4*length(mrk.pos)/5));
    
    mrk_test = mrk_selectEvents(mrk, floor(length(mrk.pos)/2)+1:length(mrk.pos));
    %     mrk_test = mrk_selectEvents(mrk, floor(4*length(mrk.pos)/5)+1:length(mrk.pos));
    
    stimulus  = stimulus_list_new( mrk );
    [response restimulus]  = response_list_new( cnt_flt, stimulus );
    stimulus_train  = stimulus_list_new( mrk_train );
    stimulus_test  = stimulus_list_new( mrk_test );
    nonstimulus = nonstimulus_list_new( mrk );
    normal = normalBrake_list_new( cnt_flt , response );
    normal_train = normal(1:1*floor(length(normal)/2));
    normal_test = normal(1*floor(length(normal)/2)+1:length(normal));
    
    restimulus_train.TargetBrake_stim = restimulus.TargetBrake_stim(1:1*floor(length(restimulus.TargetBrake_stim)/2));
    restimulus_train.NontargetBrakeOn_stim = restimulus.NontargetBrakeOn_stim(1:1*floor(length(restimulus.NontargetBrakeOn_stim)/2));
    restimulus_train.NontargetBrakeOff_stim = restimulus.NontargetBrakeOff_stim(1:1*floor(length(restimulus.NontargetBrakeOff_stim)/2));
    restimulus_train.NontargetLongBrakeOn_stim = restimulus.NontargetLongBrakeOn_stim(1:1*floor(length(restimulus.NontargetLongBrakeOn_stim)/2));
    restimulus_train.NontargetLongBrakeOff_stim = restimulus.NontargetLongBrakeOff_stim(1:1*floor(length(restimulus.NontargetLongBrakeOff_stim)/2));
    restimulus_train.Right_stim = restimulus.Right_stim(1:1*floor(length(restimulus.Right_stim)/2));
    restimulus_train.Left_stim = restimulus.Left_stim(1:1*floor(length(restimulus.Left_stim)/2));
    restimulus_train.Human_stim = restimulus.Human_stim(1:1*floor(length(restimulus.Human_stim)/2));
    
    restimulus_test.TargetBrake_stim = restimulus.TargetBrake_stim(1*floor(length(restimulus.TargetBrake_stim)/2)+1:length(restimulus.TargetBrake_stim));
    restimulus_test.NontargetBrakeOn_stim = restimulus.NontargetBrakeOn_stim(1*floor(length(restimulus.NontargetBrakeOn_stim)/2)+1:length(restimulus.NontargetBrakeOn_stim));
    restimulus_test.NontargetBrakeOff_stim = restimulus.NontargetBrakeOff_stim(1*floor(length(restimulus.NontargetBrakeOff_stim)/2)+1:length(restimulus.NontargetBrakeOff_stim));
    restimulus_test.NontargetLongBrakeOn_stim = restimulus.NontargetLongBrakeOn_stim(1*floor(length(restimulus.NontargetLongBrakeOn_stim)/2)+1:length(restimulus.NontargetLongBrakeOn_stim));
    restimulus_test.NontargetLongBrakeOff_stim = restimulus.NontargetLongBrakeOff_stim(1*floor(length(restimulus.NontargetLongBrakeOff_stim)/2)+1:length(restimulus.NontargetLongBrakeOff_stim));
    restimulus_test.Right_stim = restimulus.Right_stim(1*floor(length(restimulus.Right_stim)/2)+1:length(restimulus.Right_stim));
    restimulus_test.Left_stim = restimulus.Left_stim(1*floor(length(restimulus.Left_stim)/2)+1:length(restimulus.Left_stim));
    restimulus_test.Human_stim = restimulus.Human_stim(1*floor(length(restimulus.Human_stim)/2)+1:length(restimulus.Human_stim));
    
    % restimulus_train.brake_stim = restimulus.brake_stim(1:floor(length(restimulus.brake_stim)/2));
    % restimulus_train.right_stim = restimulus.right_stim(1:floor(length(restimulus.right_stim)/2));
    % restimulus_train.human_stim = restimulus.human_stim(1:floor(length(restimulus.human_stim)/2));
    % restimulus_test.brake_stim = restimulus.brake_stim(floor(length(restimulus.brake_stim)/2)+1:length(restimulus.brake_stim));
    % restimulus_test.right_stim = restimulus.right_stim(floor(length(restimulus.right_stim)/2)+1:length(restimulus.right_stim));
    % restimulus_test.human_stim = restimulus.human_stim(floor(length(restimulus.human_stim)/2)+1:length(restimulus.human_stim));
    
    % response_train.brake_stim = response.brake_stim(1:2*floor(length(response.brake_stim)/3));
    % response_train.right_stim = response.right_stim(1:2*floor(length(response.right_stim)/3));
    % response_train.human_stim = response.human_stim(1:2*floor(length(response.human_stim)/3));
    % response_test.brake_stim = response.brake_stim(2*floor(length(response.brake_stim)/3)+1:length(response.brake_stim));
    % response_test.right_stim = response.right_stim(2*floor(length(response.right_stim)/3)+1:length(response.right_stim));
    % response_test.human_stim = response.human_stim(2*floor(length(response.human_stim)/3)+1:length(response.human_stim));
    
    
    ival1 = [4200];
    
    %% Set the window size
    
    for i=1:71
        %  ival(i,:) = [-2700 -1200] + (i-1)*(20);
        ival(i,:) = [-1700 -200] + (i-1)*(20);
        ival2(i,:) = [-2350 -850] + (i-1)*(20);
    end
    
    
    
    %% Nontarget segmentation (epo)
    nonTarget = nontargetSegmentation_car_new(nonstimulus,normal,cnt_flt, mrk,ival(i,:));
    %     nontarget = nonTarget.nonstim;
    nontarget_train = nonTarget(:,1:floor(size(nonTarget,2)/2),:);
    %     epo_nontarget_erd.x = nonTarget(:,:,1:ind-3);
    %     epo_nontarget_erd.x = permute(epo_nontarget_erd.x, [1 3 2]);
    %     epo_nontarget_erd.t = linspace(ival(i,1),ival(i,2),size(nontarget_train,1));
    %     epo_nontarget_erd.clab = cnt_flt.clab(1,1:ind-3);
    %     epo_nontarget_erd.fs = cnt_flt.fs;
    %     epo_nontarget_erd.title = cnt_flt.title;
    % nontarget_train = nontarget(:,1:100,:);
    nontarget_test = nonTarget(:,floor(size(nonTarget,2)/2)+1:end,:);
    %     epo_nontarget_erd_test.x = nontarget_test(:,:,1:ind-3);
    %     epo_nontarget_erd_test.x = permute(epo_nontarget_erd_test.x, [1 3 2]);
    %     epo_nontarget_erd_test.t = linspace(ival(i,1),ival(i,2),size(nontarget_test,1));
    %     epo_nontarget_erd_test.clab = cnt_flt.clab(1,1:ind-3);
    %     epo_nontarget_erd_test.fs = cnt_flt.fs;
    % nontarget_test = nontarget(:,201:400,:);
    
    
    %% Classification
    for i=1:71
        
        %% Frequency band selection and filtering
        cnt_.fs = cnt_flt.fs;
        cnt_.title = cnt_flt.title;
        cnt_.file = cnt_flt.file;
        cnt_.x = cnt_flt.x(:,1:ind-3);
        cnt_.clab = cnt_flt.clab(1,1:ind-3);
        cnt_lap = proc_laplacian(cnt_);
        [filt_b, filt_a]= butter(5, [5 35]/cnt.fs*2);
        cnt_flt1= proc_filt(cnt_lap, filt_b, filt_a);
        csp_ival= select_timeival(cnt_flt1, mrk, 'do_laplace',0, 'channelwise', 1, 'start_ival', ival(i,:), 'max_ival', ival(i,:));
        clear cnt_flt1
        csp_band= select_bandnarrow(cnt_lap, mrk, csp_ival, 'do_laplace',0);
        clear cnt_lap
        [filt_b,filt_a]= butter(5, csp_band/cnt.fs*2);
        cnt_erd= proc_filt(cnt_, filt_b, filt_a);
        %     csp_ival = select_timeival(cnt_erd, mrk_train,'start_ival', ival(i,:), 'max_ival', ival(i,:) );
        
        %% Nontarget segmentation (erd)
        %         clear filt_nontarget_erd filt_nontarget_erd_test nontarget_erd nontarget_erd_train nontarget_erd_test
        %
        %         filt_nontarget_erd = proc_filtButter(epo_nontarget_erd, 5, csp_band);
        %         filt_nontarget_erd = proc_normalize(filt_nontarget_erd);
        %         nontarget_erd = permute(filt_nontarget_erd.x, [1 3 2]);
        %         nontarget_erd_train = nontarget_erd(:,1:floor(size(nontarget_erd,2)/2),:);
        %         nontarget_erd_test = nontarget_erd(:,floor(size(nontarget_erd,2)/2)+1:end,:);
        
        %         filt_nontarget_erd_test = proc_filt(epo_nontarget_erd_test,filt_b,filt_a);
        %         nontarget_erd_test = permute(filt_nontarget_erd_test.x,[1 3 2]);
        
        %         if i==1
        clear nonTarget_erd nontarget_erd_train nontarget_erd_test
        nonTarget_erd = nontargetSegmentation_car_new(nonstimulus,normal,cnt_erd, mrk,ival(end,:));
        nontarget_erd_train = nonTarget_erd(:,1:floor(size(nonTarget_erd,2)/2),:);
        Nontarget_erd_train{s,i} = nontarget_erd_train;
        %             nontarget_train = nontarget(:,1:100,:);
        nontarget_erd_test = nonTarget_erd(:,floor(size(nonTarget_erd,2)/2)+1:end,:);
        Nontarget_erd_test{s,i} = nontarget_erd_test;
        %         else
        %             clear nontarget_erd_train nontarget_erd_test nonTarget_erd;
        %             nonTarget_erd = nonTarget_erd_per;
        %             nontarget_erd_train = nonTarget_erd.nonstim(:,1:floor(size(nonTarget_erd.nonstim,2)/2),:);
        %             % nontarget_train = nontarget(:,1:100,:);
        %             nontarget_erd_test = nonTarget_erd.nonstim(:,floor(size(nonTarget_erd.nonstim,2)/2)+1:end,:);
        %
        %             clear nontarget_train nontarget_test;
        %             nontarget_train = nonTarget_per.nonstim(:,1:floor(size(nonTarget_per.nonstim,2)/2),:);
        %             nontarget_test = nonTarget_per.nonstim(:,floor(size(nonTarget_per.nonstim,2)/2)+1:end,:);
        %         end
        %% Exception process
        a(s) = size(restimulus_train.NontargetBrakeOn_stim,2);
        b(s) = size(restimulus_train.NontargetLongBrakeOn_stim,2);
        if a(s) == 0
            restimulus_train.NontargetBrakeOn_stim = [1701];
        end
        if b(s) == 0
            restimulus_train.NontargetLongBrakeOn_stim = [1701];
        end
        
        
        %% Make epo_train feature
        % target = targetSegmentation_car_new(stimulus_train,nonstimulus,cnt_flt,ival(i,:));
        
        target = targetSegmentation_car_new(restimulus_train,nonstimulus,cnt_flt,ival(i,:),ival1);
        [normalBrake target] = normalBrake_Segmentation_car_new(normal_train,target,cnt_flt,ival2(i,:));
        nonTargetBrake = nontargetBrakeSegmentation(restimulus_train,nonstimulus,cnt_flt,ival(i,:),ival1);
        
        % target = targetSegmentation_car_new(response_train,nonstimulus,cnt_flt,ival(i,:));
        epo.fs = cnt.fs;
        epo.t= linspace(ival(i,1),ival(i,2),size(nontarget_train,1));
        epo.clab = cnt_flt.clab;
        %  epo.clab = epo.clab(1,1:ind-1);
        epo.className={'SharpBraking','SoftBraking','NoBraking'};
        epo.x = cat(2,target.target_TargetBrake,target.target_Right,target.target_Left, target.target_Human,...
            target.target_NontargetBrakeOn, target.target_NontargetLongBrakeOn,target.normalBrake,...
            nonTargetBrake.NontargetBrakeOff,nonTargetBrake.NontargetLongBrakeOff,nontarget_train);
        epo.x = permute(epo.x,[1 3 2]);
        epo.y = zeros(3,size(epo.x,3));
        index1 = size(target.target_TargetBrake,2) + size(target.target_Right,2) + size(target.target_Left,2) + size(target.target_Human,2);
        index2 = size(target.target_NontargetBrakeOn,2) + size(target.target_NontargetLongBrakeOn,2) + size(target.normalBrake,2);
        index3 = size(nonTargetBrake.NontargetBrakeOff,2) + size(nonTargetBrake.NontargetLongBrakeOff,2) + size(nontarget_train,2);
        epo.y(1,1:index1) = 1;
        epo.y(2,index1+1:index1 + index2) = 1;
        epo.y(3,index1+index2+1:index1+index2+index3) = 1;
        
        epo_eeg = epo;
        epo_emg = epo;
        epo_beh = epo;
        epo_eeg.x = epo.x(:,1:ind-3,:);
        epo_eeg.clab = epo.clab(1,1:ind-3);
        epo_emg.x = epo.x(:,ind,:);
        epo_emg.clab = epo.clab(1,ind);
        epo_beh.x = epo.x(:,ind+2,:);
        epo_beh.clab = epo.clab(1,ind+2);
        
        %% SharpNoBrake
        epo_SharpNoBrake_eeg = proc_selectClasses(epo_eeg,'not',{'SoftBraking'});
        epo_SharpNoBrake_emg = proc_selectClasses(epo_emg,'not',{'SoftBraking'});
        epo_SharpNoBrake_beh = proc_selectClasses(epo_beh,'not',{'SoftBraking'});
        
        epo_SharpNoBrake_eeg = proc_baseline(epo_SharpNoBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_emg = proc_baseline(epo_SharpNoBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_beh = proc_baseline(epo_SharpNoBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SharpNoBrake_eeg = proc_selectIval(epo_SharpNoBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpNoBrake_emg = proc_selectIval(epo_SharpNoBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpNoBrake_beh = proc_selectIval(epo_SharpNoBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SharpNoBrake_eeg_r = proc_r_square_signed(epo_SharpNoBrake_eeg);
        epo_SharpNoBrake_emg_r = proc_r_square_signed(epo_SharpNoBrake_emg);
        epo_SharpNoBrake_beh_r = proc_r_square_signed(epo_SharpNoBrake_beh);
        
        ival_cfy_epo_SharpNoBrake_eeg = select_time_intervals(epo_SharpNoBrake_eeg_r, 'nIvals', 10);
        ival_cfy_epo_SharpNoBrake_emg = select_time_intervals(epo_SharpNoBrake_emg_r, 'nIvals', 10);
        ival_cfy_epo_SharpNoBrake_beh = select_time_intervals(epo_SharpNoBrake_beh_r, 'nIvals', 10);
        
        
        epo_SharpNoBrake_eeg_ = proc_jumpingMeans(epo_SharpNoBrake_eeg, ival_cfy_epo_SharpNoBrake_eeg);
        epo_SharpNoBrake_emg_ = proc_jumpingMeans(epo_SharpNoBrake_emg, ival_cfy_epo_SharpNoBrake_emg);
        epo_SharpNoBrake_beh_ = proc_jumpingMeans(epo_SharpNoBrake_beh, ival_cfy_epo_SharpNoBrake_beh);
        
        epo_SharpNoBrake_eeg = proc_flaten(epo_SharpNoBrake_eeg_);
        epo_SharpNoBrake_emg = proc_flaten(epo_SharpNoBrake_emg_);
        epo_SharpNoBrake_beh = proc_flaten(epo_SharpNoBrake_beh_);
        
        %% SoftNoBrake
        epo_SoftNoBrake_eeg = proc_selectClasses(epo_eeg,'not',{'SharpBraking'});
        epo_SoftNoBrake_emg = proc_selectClasses(epo_emg,'not',{'SharpBraking'});
        epo_SoftNoBrake_beh = proc_selectClasses(epo_beh,'not',{'SharpBraking'});
        
        epo_SoftNoBrake_eeg = proc_baseline(epo_SoftNoBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_emg = proc_baseline(epo_SoftNoBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_beh = proc_baseline(epo_SoftNoBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SoftNoBrake_eeg = proc_selectIval(epo_SoftNoBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SoftNoBrake_emg = proc_selectIval(epo_SoftNoBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SoftNoBrake_beh = proc_selectIval(epo_SoftNoBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SoftNoBrake_eeg_r = proc_r_square_signed(epo_SoftNoBrake_eeg);
        epo_SoftNoBrake_emg_r = proc_r_square_signed(epo_SoftNoBrake_emg);
        epo_SoftNoBrake_beh_r = proc_r_square_signed(epo_SoftNoBrake_beh);
        
        ival_cfy_epo_SoftNoBrake_eeg = select_time_intervals(epo_SoftNoBrake_eeg_r, 'nIvals', 10);
        ival_cfy_epo_SoftNoBrake_emg = select_time_intervals(epo_SoftNoBrake_emg_r, 'nIvals', 10);
        ival_cfy_epo_SoftNoBrake_beh = select_time_intervals(epo_SoftNoBrake_beh_r, 'nIvals', 10);
        
        
        epo_SoftNoBrake_eeg_ = proc_jumpingMeans(epo_SoftNoBrake_eeg, ival_cfy_epo_SoftNoBrake_eeg);
        epo_SoftNoBrake_emg_ = proc_jumpingMeans(epo_SoftNoBrake_emg, ival_cfy_epo_SoftNoBrake_emg);
        epo_SoftNoBrake_beh_ = proc_jumpingMeans(epo_SoftNoBrake_beh, ival_cfy_epo_SoftNoBrake_beh);
        
        epo_SoftNoBrake_eeg = proc_flaten(epo_SoftNoBrake_eeg_);
        epo_SoftNoBrake_emg = proc_flaten(epo_SoftNoBrake_emg_);
        epo_SoftNoBrake_beh = proc_flaten(epo_SoftNoBrake_beh_);
        
        %% SharpSoftBrake
        epo_SharpSoftBrake_eeg = proc_selectClasses(epo_eeg,'not',{'NoBraking'});
        epo_SharpSoftBrake_emg = proc_selectClasses(epo_emg,'not',{'NoBraking'});
        epo_SharpSoftBrake_beh = proc_selectClasses(epo_beh,'not',{'NoBraking'});
        
        epo_SharpSoftBrake_eeg = proc_baseline(epo_SharpSoftBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_emg = proc_baseline(epo_SharpSoftBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_beh = proc_baseline(epo_SharpSoftBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SharpSoftBrake_eeg = proc_selectIval(epo_SharpSoftBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpSoftBrake_emg = proc_selectIval(epo_SharpSoftBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpSoftBrake_beh = proc_selectIval(epo_SharpSoftBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SharpSoftBrake_eeg_r = proc_r_square_signed(epo_SharpSoftBrake_eeg);
        epo_SharpSoftBrake_emg_r = proc_r_square_signed(epo_SharpSoftBrake_emg);
        epo_SharpSoftBrake_beh_r = proc_r_square_signed(epo_SharpSoftBrake_beh);
        
        ival_cfy_epo_SharpSoftBrake_eeg = select_time_intervals(epo_SharpSoftBrake_eeg_r, 'nIvals', 10);
        ival_cfy_epo_SharpSoftBrake_emg = select_time_intervals(epo_SharpSoftBrake_emg_r, 'nIvals', 10);
        ival_cfy_epo_SharpSoftBrake_beh = select_time_intervals(epo_SharpSoftBrake_beh_r, 'nIvals', 10);
        
        
        epo_SharpSoftBrake_eeg_ = proc_jumpingMeans(epo_SharpSoftBrake_eeg, ival_cfy_epo_SharpSoftBrake_eeg);
        epo_SharpSoftBrake_emg_ = proc_jumpingMeans(epo_SharpSoftBrake_emg, ival_cfy_epo_SharpSoftBrake_emg);
        epo_SharpSoftBrake_beh_ = proc_jumpingMeans(epo_SharpSoftBrake_beh, ival_cfy_epo_SharpSoftBrake_beh);
        
        epo_SharpSoftBrake_eeg = proc_flaten(epo_SharpSoftBrake_eeg_);
        epo_SharpSoftBrake_emg = proc_flaten(epo_SharpSoftBrake_emg_);
        epo_SharpSoftBrake_beh = proc_flaten(epo_SharpSoftBrake_beh_);
        
        
        %% Make erd_train feature
        
        
        target_erd = targetSegmentation_car_new(restimulus_train,nonstimulus,cnt_erd,ival(i,:),ival1);
        [normalBrake_erd target_erd] = normalBrake_Segmentation_car_new(normal_train,target_erd,cnt_erd,ival2(i,:));
        %         target_erd.normalBrake = target_erd.normalBrake(cnt.fs*(300/1000) + cnt.fs*(csp_ival(1,1)/1000) : cnt.fs*(300/1000) + cnt.fs*(csp_ival(1,2)/1000),:,:);
        nonTarget_erd_brake = nontargetBrakeSegmentation(restimulus_train,nonstimulus,cnt_erd,ival(i,:),ival1);
        %         N = round(cnt.fs*((csp_ival(1,2) - csp_ival(1,1))/1000));
        %         nontarget_erd_train = nontarget_erd_orig_train(1:(N+1),:,:);
        
        % target = targetSegmentation_car_new(response_train,nonstimulus,cnt_flt,ival(i,:));
        epo_erd.fs = cnt.fs;
        epo_erd.t= linspace(ival(i,1),ival(i,2),size(nontarget_erd_train,1));
        epo_erd.clab = cnt_erd.clab;
        %  epo_erd.clab = epo_erd.clab(1,1:ind-1);
        epo_erd.className={'SharpBraking','SoftBraking','NoBraking'};
        epo_erd.x = cat(2,target_erd.target_TargetBrake,target_erd.target_Right,target_erd.target_Left, target_erd.target_Human,...
            target_erd.target_NontargetBrakeOn, target_erd.target_NontargetLongBrakeOn,target_erd.normalBrake,...
            nonTarget_erd_brake.NontargetBrakeOff,nonTarget_erd_brake.NontargetLongBrakeOff,nontarget_erd_train);
        epo_erd.x = permute(epo_erd.x,[1 3 2]);
        epo_erd.y = zeros(3,size(epo_erd.x,3));
        index1 = size(target_erd.target_TargetBrake,2) + size(target_erd.target_Right,2) + size(target_erd.target_Left,2) + size(target_erd.target_Human,2);
        index2 = size(target_erd.target_NontargetBrakeOn,2) + size(target_erd.target_NontargetLongBrakeOn,2) + size(target_erd.normalBrake,2);
        index3 = size(nonTarget_erd_brake.NontargetBrakeOff,2) + size(nonTarget_erd_brake.NontargetLongBrakeOff,2) + size(nontarget_erd_train,2);
        epo_erd.y(1,1:index1) = 1;
        epo_erd.y(2,index1+1:index1 + index2) = 1;
        epo_erd.y(3,index1+index2+1:index1+index2+index3) = 1;
        
        %% SharpNoBrake
        epo_SharpNoBrake_erd = proc_selectClasses(epo_erd,'not',{'SoftBraking'});
        %         epo_SharpNoBrake_erd = proc_baseline(epo_SharpNoBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_erd = proc_selectIval(epo_SharpNoBrake_erd, [ival(i,1)+100 ival(i,2)]);
        [epo_SharpNoBrake_erd csp_SharpNoBrake_w] = proc_csp3(epo_SharpNoBrake_erd, 3);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SharpNoBrake_erd = proc_variance(epo_SharpNoBrake_erd);
        epo_SharpNoBrake_erd = proc_logarithm(epo_SharpNoBrake_erd);
        epo_SharpNoBrake_erd = proc_flaten(epo_SharpNoBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SharpNoBrake_eeg = proc_catFeatures(epo_SharpNoBrake_eeg, epo_SharpNoBrake_erd);
        [fv_SharpNoBrake_eeg_train, opt_SharpNoBrake_eeg] = proc_normalize(fv_SharpNoBrake_eeg);
        [fv_SharpNoBrake_emg_train, opt_SharpNoBrake_emg] = proc_normalize(epo_SharpNoBrake_emg);
        [fv_SharpNoBrake_beh_train, opt_SharpNoBrake_beh] = proc_normalize(epo_SharpNoBrake_beh);
        
        
        %% SoftNoBrake
        
        epo_SoftNoBrake_erd = proc_selectClasses(epo_erd,'not',{'SharpBraking'});
        %         epo_SoftNoBrake_erd = proc_baseline(epo_SoftNoBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_erd = proc_selectIval(epo_SoftNoBrake_erd, [ival(i,1)+100 ival(i,2)]);
        [epo_SoftNoBrake_erd csp_SoftNoBrake_w] = proc_csp3(epo_SoftNoBrake_erd, 3);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SoftNoBrake_erd = proc_variance(epo_SoftNoBrake_erd);
        epo_SoftNoBrake_erd = proc_logarithm(epo_SoftNoBrake_erd);
        epo_SoftNoBrake_erd = proc_flaten(epo_SoftNoBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SoftNoBrake_eeg = proc_catFeatures(epo_SoftNoBrake_eeg, epo_SoftNoBrake_erd);
        [fv_SoftNoBrake_eeg_train, opt_SoftNoBrake_eeg] = proc_normalize(fv_SoftNoBrake_eeg);
        [fv_SoftNoBrake_emg_train, opt_SoftNoBrake_emg] = proc_normalize(epo_SoftNoBrake_emg);
        [fv_SoftNoBrake_beh_train, opt_SoftNoBrake_beh] = proc_normalize(epo_SoftNoBrake_beh);
        
        
        %% SharpSoftBrake
        
        epo_SharpSoftBrake_erd = proc_selectClasses(epo_erd,'not',{'NoBraking'});
        %         epo_SharpSoftBrake_erd = proc_baseline(epo_SharpSoftBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_erd = proc_selectIval(epo_SharpSoftBrake_erd, [ival(i,1)+100 ival(i,2)]);
        [epo_SharpSoftBrake_erd csp_SharpSoftBrake_w] = proc_csp3(epo_SharpSoftBrake_erd, 3);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SharpSoftBrake_erd = proc_variance(epo_SharpSoftBrake_erd);
        epo_SharpSoftBrake_erd = proc_logarithm(epo_SharpSoftBrake_erd);
        epo_SharpSoftBrake_erd = proc_flaten(epo_SharpSoftBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SharpSoftBrake_eeg = proc_catFeatures(epo_SharpSoftBrake_eeg, epo_SharpSoftBrake_erd);
        [fv_SharpSoftBrake_eeg_train, opt_SharpSoftBrake_eeg] = proc_normalize(fv_SharpSoftBrake_eeg);
        [fv_SharpSoftBrake_emg_train, opt_SharpSoftBrake_emg] = proc_normalize(epo_SharpSoftBrake_emg);
        [fv_SharpSoftBrake_beh_train, opt_SharpSoftBrake_beh] = proc_normalize(epo_SharpSoftBrake_beh);
        
        
        
        C_SharpNoBrake_eeg= trainClassifier(fv_SharpNoBrake_eeg_train,{'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SharpNoBrake_emg= trainClassifier(fv_SharpNoBrake_emg_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SharpNoBrake_beh= trainClassifier(fv_SharpNoBrake_beh_train,{'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        
        C_SoftNoBrake_eeg= trainClassifier(fv_SoftNoBrake_eeg_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SoftNoBrake_emg= trainClassifier(fv_SoftNoBrake_emg_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SoftNoBrake_beh= trainClassifier(fv_SoftNoBrake_beh_train,{'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        
        
        C_SharpSoftBrake_eeg= trainClassifier(fv_SharpSoftBrake_eeg_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SharpSoftBrake_emg= trainClassifier(fv_SharpSoftBrake_emg_train,{'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        C_SharpSoftBrake_beh= trainClassifier(fv_SharpSoftBrake_beh_train, {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
            'store_cov', 1, 'store_invcov', 1, 'scaling', 1});
        
        
        clear target target_erd nonTarget_erd_brake nonTargetBrake normalBrake normalBrake_erd epo_SharpNoBrake_eeg epo_SharpNoBrake_emg epo_SharpNoBrake_beh...
            epo_SharpNoBrake_eeg_r epo_SharpNoBrake_emg_r epo_SharpNoBrake_beh_r epo epo_erd epo_eeg epo_emg epo_beh ...
            epo_SoftNoBrake_eeg epo_SoftNoBrake_emg epo_SoftNoBrake_beh  epo_SoftNoBrake_eeg_r epo_SoftNoBrake_emg_r epo_SoftNoBrake_beh_r  ...
            epo_SoftNoBrake_eeg_ epo_SoftNoBrake_emg_ epo_SoftNoBrake_beh_ epo_SharpNoBrake_eeg_ epo_SharpNoBrake_emg_ epo_SharpNoBrake_beh_ ...
            epo_SharpSoftBrake_eeg epo_SharpSoftBrake_emg epo_SharpSoftBrake_beh epo_SharpSoftBrake_eeg_r epo_SharpSoftBrake_emg_r epo_SharpSoftBrake_beh_r ...
            epo_SharpSoftBrake_eeg_ epo_SharpSoftBrake_emg_ epo_SharpSoftBrake_beh_ epo_SharpNoBrake_erd fv_SharpNoBrake_eeg epo_SoftNoBrake_erd ...
            fv_SoftNoBrake_eeg epo_SharpSoftBrake_erd fv_SharpSoftBrake_eeg fv_SharpNoBrake_eeg_train fv_SharpNoBrake_emg_train fv_SharpNoBrake_beh_train ...
            fv_SoftNoBrake_eeg_train fv_SoftNoBrake_emg_train fv_SoftNoBrake_beh_train fv_SharpSoftBrake_eeg_train fv_SharpSoftBrake_emg_train fv_SharpSoftBrake_beh_train
        
        
        %         nonTarget.nonstim = zeros(1, 1);
        %         nonTarget_erd.nonstim = zeros(1, 1);
        %
        
        %% Make epo_test feature
        c(s) = size(restimulus_test.NontargetBrakeOn_stim,2);
        d(s) = size(restimulus_test.NontargetLongBrakeOn_stim,2);
        if c(s) == 0
            restimulus_test.NontargetBrakeOn_stim = [1701];
        end
        if d(s) == 0
            restimulus_test.NontargetLongBrakeOn_stim = [1701];
        end
        
        target = targetSegmentation_car_new(restimulus_test,nonstimulus,cnt_flt,ival(i,:),ival1);
        [normalBrake target] = normalBrake_Segmentation_car_new(normal_test,target,cnt_flt,ival2(i,:));
        nonTargetBrake = nontargetBrakeSegmentation(restimulus_test,nonstimulus,cnt_flt,ival(i,:),ival1);
        
        % target = targetSegmentation_car_new(response_test,nonstimulus,cnt_flt,ival(i,:));
        epo.fs = cnt.fs;
        epo.t= linspace(ival(i,1),ival(i,2),size(nontarget_test,1));
        epo.clab = cnt_flt.clab;
        %  epo.clab = epo.clab(1,1:ind-1);
        epo.className={'SharpBraking','SoftBraking','NoBraking'};
        epo.x = cat(2,target.target_TargetBrake,target.target_Right,target.target_Left, target.target_Human,...
            target.target_NontargetBrakeOn, target.target_NontargetLongBrakeOn,target.normalBrake,...
            nonTargetBrake.NontargetBrakeOff,nonTargetBrake.NontargetLongBrakeOff,nontarget_test);
        epo.x = permute(epo.x,[1 3 2]);
        epo.y = zeros(3,size(epo.x,3));
        index1 = size(target.target_TargetBrake,2) + size(target.target_Right,2) + size(target.target_Left,2) + size(target.target_Human,2);
        index2 = size(target.target_NontargetBrakeOn,2) + size(target.target_NontargetLongBrakeOn,2) + size(target.normalBrake,2);
        index3 = size(nonTargetBrake.NontargetBrakeOff,2) + size(nonTargetBrake.NontargetLongBrakeOff,2) + size(nontarget_test,2);
        epo.y(1,1:index1) = 1;
        epo.y(2,index1+1:index1 + index2) = 1;
        epo.y(3,index1+index2+1:index1+index2+index3) = 1;
        
        epo_eeg = epo;
        epo_emg = epo;
        epo_beh = epo;
        epo_eeg.x = epo.x(:,1:ind-3,:);
        epo_eeg.clab = epo.clab(1,1:ind-3);
        epo_emg.x = epo.x(:,ind,:);
        epo_emg.clab = epo.clab(1,ind);
        epo_beh.x = epo.x(:,ind+2,:);
        epo_beh.clab = epo.clab(1,ind+2);
        
        %% SharpNoBrake
        epo_SharpNoBrake_eeg = proc_selectClasses(epo_eeg,'not',{'SoftBraking'});
        epo_SharpNoBrake_emg = proc_selectClasses(epo_emg,'not',{'SoftBraking'});
        epo_SharpNoBrake_beh = proc_selectClasses(epo_beh,'not',{'SoftBraking'});
        
        epo_SharpNoBrake_eeg = proc_baseline(epo_SharpNoBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_emg = proc_baseline(epo_SharpNoBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_beh = proc_baseline(epo_SharpNoBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SharpNoBrake_eeg = proc_selectIval(epo_SharpNoBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpNoBrake_emg = proc_selectIval(epo_SharpNoBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpNoBrake_beh = proc_selectIval(epo_SharpNoBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SharpNoBrake_eeg_ = proc_jumpingMeans(epo_SharpNoBrake_eeg, ival_cfy_epo_SharpNoBrake_eeg);
        epo_SharpNoBrake_emg_ = proc_jumpingMeans(epo_SharpNoBrake_emg, ival_cfy_epo_SharpNoBrake_emg);
        epo_SharpNoBrake_beh_ = proc_jumpingMeans(epo_SharpNoBrake_beh, ival_cfy_epo_SharpNoBrake_beh);
        
        epo_SharpNoBrake_eeg = proc_flaten(epo_SharpNoBrake_eeg_);
        epo_SharpNoBrake_emg = proc_flaten(epo_SharpNoBrake_emg_);
        epo_SharpNoBrake_beh = proc_flaten(epo_SharpNoBrake_beh_);
        
        
        %% SoftNoBrake
        epo_SoftNoBrake_eeg = proc_selectClasses(epo_eeg,'not',{'SharpBraking'});
        epo_SoftNoBrake_emg = proc_selectClasses(epo_emg,'not',{'SharpBraking'});
        epo_SoftNoBrake_beh = proc_selectClasses(epo_beh,'not',{'SharpBraking'});
        
        epo_SoftNoBrake_eeg = proc_baseline(epo_SoftNoBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_emg = proc_baseline(epo_SoftNoBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_beh = proc_baseline(epo_SoftNoBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SoftNoBrake_eeg = proc_selectIval(epo_SoftNoBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SoftNoBrake_emg = proc_selectIval(epo_SoftNoBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SoftNoBrake_beh = proc_selectIval(epo_SoftNoBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SoftNoBrake_eeg_ = proc_jumpingMeans(epo_SoftNoBrake_eeg, ival_cfy_epo_SoftNoBrake_eeg);
        epo_SoftNoBrake_emg_ = proc_jumpingMeans(epo_SoftNoBrake_emg, ival_cfy_epo_SoftNoBrake_emg);
        epo_SoftNoBrake_beh_ = proc_jumpingMeans(epo_SoftNoBrake_beh, ival_cfy_epo_SoftNoBrake_beh);
        
        epo_SoftNoBrake_eeg = proc_flaten(epo_SoftNoBrake_eeg_);
        epo_SoftNoBrake_emg = proc_flaten(epo_SoftNoBrake_emg_);
        epo_SoftNoBrake_beh = proc_flaten(epo_SoftNoBrake_beh_);
        
        
        %% SharpSoftBrake
        epo_SharpSoftBrake_eeg = proc_selectClasses(epo_eeg,'not',{'NoBraking'});
        epo_SharpSoftBrake_emg = proc_selectClasses(epo_emg,'not',{'NoBraking'});
        epo_SharpSoftBrake_beh = proc_selectClasses(epo_beh,'not',{'NoBraking'});
        
        epo_SharpSoftBrake_eeg = proc_baseline(epo_SharpSoftBrake_eeg,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_emg = proc_baseline(epo_SharpSoftBrake_emg,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_beh = proc_baseline(epo_SharpSoftBrake_beh,[ival(i,1) ival(i,1)+100]);
        
        epo_SharpSoftBrake_eeg = proc_selectIval(epo_SharpSoftBrake_eeg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpSoftBrake_emg = proc_selectIval(epo_SharpSoftBrake_emg, [ival(i,1)+100 ival(i,2)]);
        epo_SharpSoftBrake_beh = proc_selectIval(epo_SharpSoftBrake_beh, [ival(i,1)+100 ival(i,2)]);
        
        epo_SharpSoftBrake_eeg_ = proc_jumpingMeans(epo_SharpSoftBrake_eeg, ival_cfy_epo_SharpSoftBrake_eeg);
        epo_SharpSoftBrake_emg_ = proc_jumpingMeans(epo_SharpSoftBrake_emg, ival_cfy_epo_SharpSoftBrake_emg);
        epo_SharpSoftBrake_beh_ = proc_jumpingMeans(epo_SharpSoftBrake_beh, ival_cfy_epo_SharpSoftBrake_beh);
        
        epo_SharpSoftBrake_eeg = proc_flaten(epo_SharpSoftBrake_eeg_);
        epo_SharpSoftBrake_emg = proc_flaten(epo_SharpSoftBrake_emg_);
        epo_SharpSoftBrake_beh = proc_flaten(epo_SharpSoftBrake_beh_);
        %% Make erd_test feature
        
        
        target_erd = targetSegmentation_car_new(restimulus_test,nonstimulus,cnt_erd,ival(i,:),ival1);
        [normalBrake_erd target_erd] = normalBrake_Segmentation_car_new(normal_test,target_erd,cnt_erd,ival2(i,:));
        %         target_erd.normalBrake = target_erd.normalBrake(cnt.fs*(300/1000) + cnt.fs*(csp_ival(1,1)/1000) : cnt.fs*(300/1000) + cnt.fs*(csp_ival(1,2)/1000),:,:);
        nonTarget_erd_brake = nontargetBrakeSegmentation(restimulus_test,nonstimulus,cnt_erd,ival(i,:),ival1);
        %         N = round(cnt.fs*((csp_ival(1,2) - csp_ival(1,1))/1000));
        %         nontarget_erd_test = nontarget_erd_orig_test(1:(N+1),:,:);
        
        % target = targetSegmentation_car_new(response_test,nonstimulus,cnt_flt,ival(i,:));
        epo_erd.fs = cnt.fs;
        epo_erd.t= linspace(ival(i,1),ival(i,2),size(nontarget_erd_test,1));
        epo_erd.clab = cnt_erd.clab;
        %  epo_erd.clab = epo_erd.clab(1,1:ind-1);
        epo_erd.className={'SharpBraking','SoftBraking','NoBraking'};
        epo_erd.x = cat(2,target_erd.target_TargetBrake,target_erd.target_Right,target_erd.target_Left, target_erd.target_Human,...
            target_erd.target_NontargetBrakeOn, target_erd.target_NontargetLongBrakeOn,target_erd.normalBrake,...
            nonTarget_erd_brake.NontargetBrakeOff,nonTarget_erd_brake.NontargetLongBrakeOff,nontarget_erd_test);
        epo_erd.x = permute(epo_erd.x,[1 3 2]);
        epo_erd.y = zeros(3,size(epo_erd.x,3));
        index1 = size(target_erd.target_TargetBrake,2) + size(target_erd.target_Right,2) + size(target_erd.target_Left,2) + size(target_erd.target_Human,2);
        index2 = size(target_erd.target_NontargetBrakeOn,2) + size(target_erd.target_NontargetLongBrakeOn,2) + size(target_erd.normalBrake,2);
        index3 = size(nonTarget_erd_brake.NontargetBrakeOff,2) + size(nonTarget_erd_brake.NontargetLongBrakeOff,2) + size(nontarget_erd_test,2);
        epo_erd.y(1,1:index1) = 1;
        epo_erd.y(2,index1+1:index1 + index2) = 1;
        epo_erd.y(3,index1+index2+1:index1+index2+index3) = 1;
        
        
        %% SharpNoBrake
        epo_SharpNoBrake_erd = proc_selectClasses(epo_erd,'not',{'SoftBraking'});
        %         epo_SharpNoBrake_erd = proc_baseline(epo_SharpNoBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SharpNoBrake_erd = proc_selectIval(epo_SharpNoBrake_erd, [ival(i,1)+100 ival(i,2)]);
        epo_SharpNoBrake_erd  = proc_linearDerivation(epo_SharpNoBrake_erd, csp_SharpNoBrake_w);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SharpNoBrake_erd = proc_variance(epo_SharpNoBrake_erd);
        epo_SharpNoBrake_erd = proc_logarithm(epo_SharpNoBrake_erd);
        epo_SharpNoBrake_erd = proc_flaten(epo_SharpNoBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SharpNoBrake_eeg = proc_catFeatures(epo_SharpNoBrake_eeg, epo_SharpNoBrake_erd);
        fv_SharpNoBrake_eeg_test = proc_normalize(fv_SharpNoBrake_eeg, opt_SharpNoBrake_eeg);
        fv_SharpNoBrake_emg_test = proc_normalize(epo_SharpNoBrake_emg, opt_SharpNoBrake_emg);
        fv_SharpNoBrake_beh_test = proc_normalize(epo_SharpNoBrake_beh, opt_SharpNoBrake_beh);
        
        out_SharpNoBrake_eeg = applyClassifier(fv_SharpNoBrake_eeg_test, 'RLDAshrink', C_SharpNoBrake_eeg);
        out_SharpNoBrake_emg = applyClassifier(fv_SharpNoBrake_emg_test, 'RLDAshrink', C_SharpNoBrake_emg);
        out_SharpNoBrake_beh = applyClassifier(fv_SharpNoBrake_beh_test, 'RLDAshrink', C_SharpNoBrake_beh);
        
        
        % out_brake= apply_separatingHyperplane(C_brake,fv_brake_test.x );
        aoc_SharpNoBrake.eeg(s,i) = AUC_new(fv_SharpNoBrake_eeg_test.y, out_SharpNoBrake_eeg);
        aoc_SharpNoBrake.emg(s,i) = AUC_new(fv_SharpNoBrake_emg_test.y, out_SharpNoBrake_emg);
        aoc_SharpNoBrake.beh(s,i) = AUC_new(fv_SharpNoBrake_beh_test.y, out_SharpNoBrake_beh);
        
        
        %% SoftNoBrake
        
        epo_SoftNoBrake_erd = proc_selectClasses(epo_erd,'not',{'SharpBraking'});
        %         epo_SoftNoBrake_erd = proc_baseline(epo_SoftNoBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SoftNoBrake_erd = proc_selectIval(epo_SoftNoBrake_erd, [ival(i,1)+100 ival(i,2)]);
        epo_SoftNoBrake_erd = proc_linearDerivation(epo_SoftNoBrake_erd, csp_SoftNoBrake_w);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SoftNoBrake_erd = proc_variance(epo_SoftNoBrake_erd);
        epo_SoftNoBrake_erd = proc_logarithm(epo_SoftNoBrake_erd);
        epo_SoftNoBrake_erd = proc_flaten(epo_SoftNoBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SoftNoBrake_eeg = proc_catFeatures(epo_SoftNoBrake_eeg, epo_SoftNoBrake_erd);
        fv_SoftNoBrake_eeg_test = proc_normalize(fv_SoftNoBrake_eeg, opt_SoftNoBrake_eeg);
        fv_SoftNoBrake_emg_test = proc_normalize(epo_SoftNoBrake_emg, opt_SoftNoBrake_emg);
        fv_SoftNoBrake_beh_test = proc_normalize(epo_SoftNoBrake_beh, opt_SoftNoBrake_beh);
        
        out_SoftNoBrake_eeg = applyClassifier(fv_SoftNoBrake_eeg_test, 'RLDAshrink', C_SoftNoBrake_eeg);
        out_SoftNoBrake_emg = applyClassifier(fv_SoftNoBrake_emg_test, 'RLDAshrink', C_SoftNoBrake_emg);
        out_SoftNoBrake_beh = applyClassifier(fv_SoftNoBrake_beh_test, 'RLDAshrink', C_SoftNoBrake_beh);
        
        
        % out_brake= apply_separatingHyperplane(C_brake,fv_brake_test.x );
        aoc_SoftNoBrake.eeg(s,i) = AUC_new(fv_SoftNoBrake_eeg_test.y, out_SoftNoBrake_eeg);
        aoc_SoftNoBrake.emg(s,i) = AUC_new(fv_SoftNoBrake_emg_test.y, out_SoftNoBrake_emg);
        aoc_SoftNoBrake.beh(s,i) = AUC_new(fv_SoftNoBrake_beh_test.y, out_SoftNoBrake_beh);
        
        
        
        %% SharpSoftBrake
        
        epo_SharpSoftBrake_erd = proc_selectClasses(epo_erd,'not',{'NoBraking'});
        %         epo_SharpSoftBrake_erd = proc_baseline(epo_SharpSoftBrake_erd,[ival(i,1) ival(i,1)+100]);
        epo_SharpSoftBrake_erd = proc_selectIval(epo_SharpSoftBrake_erd, [ival(i,1)+100 ival(i,2)]);
        epo_SharpSoftBrake_erd = proc_linearDerivation(epo_SharpSoftBrake_erd, csp_SharpSoftBrake_w);
        %     epo_erd = proc_envelope(epo_erd);
        epo_SharpSoftBrake_erd = proc_variance(epo_SharpSoftBrake_erd);
        epo_SharpSoftBrake_erd = proc_logarithm(epo_SharpSoftBrake_erd);
        epo_SharpSoftBrake_erd = proc_flaten(epo_SharpSoftBrake_erd);
        %     epo_erd = proc_baseline(epo_erd,[ival(i,1) ival(i,1)+100]);
        %     epo_erd = proc_selectIval(epo_erd, [ival(i,1)+100 ival(i,2)]);
        %     epo_erd_r = proc_r_square_signed(epo_erd);
        %     ival_cfy_epo_erd = select_time_intervals(epo_erd_r, 'nIvals', 5);
        %     epo_erd_ = proc_jumpingMeans(epo_erd, ival_cfy_epo_erd);
        
        %     epo_erd_train = proc_flaten(epo_erd_);
        
        fv_SharpSoftBrake_eeg = proc_catFeatures(epo_SharpSoftBrake_eeg, epo_SharpSoftBrake_erd);
        fv_SharpSoftBrake_eeg_test = proc_normalize(fv_SharpSoftBrake_eeg, opt_SharpSoftBrake_eeg);
        fv_SharpSoftBrake_emg_test = proc_normalize(epo_SharpSoftBrake_emg, opt_SharpSoftBrake_emg);
        fv_SharpSoftBrake_beh_test = proc_normalize(epo_SharpSoftBrake_beh, opt_SharpSoftBrake_beh);
        
        
        out_SharpSoftBrake_eeg = applyClassifier(fv_SharpSoftBrake_eeg_test, 'RLDAshrink', C_SharpSoftBrake_eeg);
        out_SharpSoftBrake_emg = applyClassifier(fv_SharpSoftBrake_emg_test, 'RLDAshrink', C_SharpSoftBrake_emg);
        out_SharpSoftBrake_beh = applyClassifier(fv_SharpSoftBrake_beh_test, 'RLDAshrink', C_SharpSoftBrake_beh);
        
        
        % out_brake= apply_separatingHyperplane(C_brake,fv_brake_test.x );
        aoc_SharpSoftBrake.eeg(s,i) = AUC_new(fv_SharpSoftBrake_eeg_test.y, out_SharpSoftBrake_eeg);
        aoc_SharpSoftBrake.emg(s,i) = AUC_new(fv_SharpSoftBrake_emg_test.y, out_SharpSoftBrake_emg);
        aoc_SharpSoftBrake.beh(s,i) = AUC_new(fv_SharpSoftBrake_beh_test.y, out_SharpSoftBrake_beh);
        
        
        
        clear target target_erd nonTargetBrake nonTarget_erd_brake normalBrake normalBrake_erd epo_SharpNoBrake_eeg epo_SharpNoBrake_emg epo_SharpNoBrake_beh...
            epo_SharpNoBrake_eeg_r epo_SharpNoBrake_emg_r epo_SharpNoBrake_beh_r epo epo_erd epo_eeg epo_emg epo_beh ...
            epo_SoftNoBrake_eeg epo_SoftNoBrake_emg epo_SoftNoBrake_beh  epo_SoftNoBrake_eeg_r epo_SoftNoBrake_emg_r epo_SoftNoBrake_beh_r  ...
            epo_SoftNoBrake_eeg_ epo_SoftNoBrake_emg_ epo_SoftNoBrake_beh_ epo_SharpNoBrake_eeg_ epo_SharpNoBrake_emg_ epo_SharpNoBrake_beh_ ...
            epo_SharpSoftBrake_eeg epo_SharpSoftBrake_emg epo_SharpSoftBrake_beh epo_SharpSoftBrake_eeg_r epo_SharpSoftBrake_emg_r epo_SharpSoftBrake_beh_r ...
            epo_SharpSoftBrake_eeg_ epo_SharpSoftBrake_emg_ epo_SharpSoftBrake_beh_ epo_SharpNoBrake_erd fv_SharpNoBrake_eeg epo_SoftNoBrake_erd ...
            fv_SoftNoBrake_eeg epo_SharpSoftBrake_erd fv_SharpSoftBrake_eeg C_SharpNoBrake_eeg C_SharpNoBrake_emg C_SharpNoBrake_beh ...
            C_SoftNoBrake_eeg C_SoftNoBrake_emg C_SoftNoBrake_beh C_SharpSoftBrake_eeg C_SharpSoftBrake_emg C_SharpSoftBrake_beh opt_SharpNoBrake_eeg ...
            opt_SharpNoBrake_emg opt_SharpNoBrake_beh opt_SoftNoBrake_eeg opt_SoftNoBrake_emg opt_SoftNoBrake_beh opt_SharpSoftBrake_eeg opt_SharpSoftBrake_emg ...
            opt_SharpSoftBrake_beh out_SharpNoBrake_eeg out_SharpNoBrake_emg out_SharpNoBrake_beh out_SoftNoBrake_eeg out_SoftNoBrake_emg out_SoftNoBrake_beh ...
            out_SharpSoftBrake_eeg out_SharpSoftBrake_emg out_SharpSoftBrake_beh fv_SharpNoBrake_eeg_test fv_SharpNoBrake_emg_test fv_SharpNoBrake_beh_test ...
            fv_SoftNoBrake_eeg_test fv_SoftNoBrake_emg_test fv_SoftNoBrake_beh_test fv_SharpSoftBrake_eeg_test fv_SharpSoftBrake_emg_test fv_SharpSoftBrake_beh_test ...
            ival_cfy_epo_SharpNoBrake_eeg ival_cfy_epo_SharpNoBrake_emg ival_cfy_epo_SharpNoBrake_beh ival_cfy_epo_SharpSoftBrake_eeg ...
            ival_cfy_epo_SharpSoftBrake_emg ival_cfy_epo_SharpSoftBrake_beh ival_cfy_epo_SoftNoBrake_eeg ival_cfy_epo_SoftNoBrake_emg ...
            ival_cfy_epo_SoftNoBrake_beh csp_SharpNoBrake_w csp_SoftNoBrake_w csp_SharpSoftBrake_w cnt_erd cnt_ csp_band csp_ival filt_a filt_b
        %         nonTarget.nonstim = zeros(1, 1);
        %         nonTarget_erd.nonstim = zeros(1, 1);
        s
        i
    end
    clear nonTarget nonTarget_erd nontarget_train nontarget_test nontarget_erd_train nontarget_erd_test cnt cnt_ cnt_1 cnt_erd cnt_flt cnt_flt_em ...
        mnt mnt1 mnt2 mnt_70 mnt_71 mrk mrk_test mrk_train nonstimulus normal normal_test normal_train response response_train ...
        response_test restimulus restimulus_train restimulus_test stimulus stimulus_train stimulus_test target target_erd y
    s
end


%% Total subject

aoc_SharpNoBrake_grand.eeg = mean(aoc_SharpNoBrake.eeg,1);
aoc_SharpNoBrake_grand.emg = mean(aoc_SharpNoBrake.emg,1);
aoc_SharpNoBrake_grand.beh = mean(aoc_SharpNoBrake.beh,1);

aoc_SoftNoBrake_grand.eeg = mean(aoc_SoftNoBrake.eeg,1);
aoc_SoftNoBrake_grand.emg = mean(aoc_SoftNoBrake.emg,1);
aoc_SoftNoBrake_grand.beh = mean(aoc_SoftNoBrake.beh,1);

aoc_SharpSoftBrake_grand.eeg = mean(aoc_SharpSoftBrake.eeg,1);
aoc_SharpSoftBrake_grand.emg = mean(aoc_SharpSoftBrake.emg,1);
aoc_SharpSoftBrake_grand.beh = mean(aoc_SharpSoftBrake.beh,1);




figure;

t= linspace(ival(end,1)+100,ival(end,2),71);
plot(t,aoc_SharpNoBrake_grand.eeg,'*-r')
hold on
plot(t,aoc_SharpNoBrake_grand.emg,'-.b')
hold on
plot(t,aoc_SharpNoBrake_grand.beh,'--c')
legend('EEG','EMG','BrakeGas')
title('Classification performance between sharply braking and no braking about for all subject')
ylabel('Accuracy');
xlabel('Time(ms)')
grid on;


figure;

% t= linspace(ival(i,1)+100,ival(i,2),71);
plot(t,aoc_SoftNoBrake_grand.eeg,'*-r')
hold on
plot(t,aoc_SoftNoBrake_grand.emg,'-.b')
hold on
plot(t,aoc_SoftNoBrake_grand.beh,'--c')
legend('EEG','EMG','BrakeGas')
title('Classification performance between softly braking and no braking about for all subject')
ylabel('Accuracy');
xlabel('Time(ms)')
grid on;

figure;

% t= linspace(ival(i,1)+100,ival(i,2),71);
plot(t,aoc_SharpSoftBrake_grand.eeg,'*-r')
hold on
plot(t,aoc_SharpSoftBrake_grand.emg,'-.b')
hold on
plot(t,aoc_SharpSoftBrake_grand.beh,'--c')
legend('EEG','EMG','BrakeGas')
title('Classification performance between sharply braking and softly braking about for all subject')
ylabel('Accuracy');
xlabel('Time(ms)')
grid on;




