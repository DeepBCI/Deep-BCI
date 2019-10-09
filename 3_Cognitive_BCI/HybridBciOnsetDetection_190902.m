% This is a code to detect subject's task onset simultaneously using EEG and NIRS
% Most of MATLAB functions are available in BBCI toolbox
% Some minor code modifications might be applied
% We do not guarantee all of functions works properly in your platform
% If you want to see more tutorials, visit BBCI toolbox (https://github.com/bbci/bbci_public)

clear all
close all
clc

%% Initial setting of the BBCI toolbox
% --- (1) specify your eeg data directory (EegMyDataDir) ------------------
EegMyDataDir = 'D:\Works\2018\Researches\1_Hybrid DNN\Data\2_Hybrid Data Analysis\EEG_Data';

% --- (2) start the BBCI toolbox ------------------------------------------
startup_bbci_toolbox('DataDir',EegMyDataDir,'TmpDir','/tmp/');
BTB.History = 0;                                                           % to aviod error for merging cnt

%% Preprocessing and Classification
 
h = waitbar(0, 'please wait ...');
sub_ind = 1:29;                                                            % Subjects' indexes (change this according to the number of your subjects)

for temp_subNum = 1:size(sub_ind, 2)
    
    %%%%% NIRS Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- (1) set parameters ----------------------------------------------
    subdir_list = {['subject ', num2str(sub_ind(temp_subNum))]}; % subject
 
    % --- (2) load nirs data ----------------------------------------------
    NirsMyDataDir = 'D:\Works\2018\Researches\1_Hybrid DNN\Data\2_Hybrid Data Analysis\NIRS_Data';
    loadDir = fullfile(NirsMyDataDir, subdir_list{1});
    cd(loadDir);
    load cnt; load mrk, load mnt; % load continous eeg signal (cnt), marker (mrk) and montage (mnt)

    % --- (3) MBLL and save post-MBLL data --------------------------------
    for idx = 1 : 6  
        filename{idx} = fullfile(subdir_list{1}, ['session',num2str(idx)]);
        cntHb{idx} = proc_BeerLambert(cnt{idx}, 'Opdist', 3, 'DPF', [5.98 7.15], 'Citation', 1);
        file_saveNIRSMatlab(filename{idx}, cntHb{idx}, mrk{idx}, mnt);
    end 

    clear cnt cntHb mrk;
 
    clear cnt_temp mrk_temp
    % --- (4) load deoxy- and oxy-hemoglobin data -------------------------
    for idx = 1 : 6
        [cnt_temp.deoxy{idx}, mrk_temp{idx}, mnt] = file_loadNIRSMatlab(filename{idx}, 'Signal','deoxy');
        [cnt_temp.oxy{idx}  , ~, ~]   = file_loadNIRSMatlab(filename{idx}, 'Signal','oxy');
    end

    % --- (5) merge cnts in each session ----------------------------------
    % for mental arithmetic: ment
    [cnt.ment.deoxy, mrk.ment.deoxy] = proc_appendCnt({cnt_temp.deoxy{2}, cnt_temp.deoxy{4}, cnt_temp.deoxy{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); % merged mental arithmetic cnts
    [cnt.ment.oxy, mrk.ment.oxy]     = proc_appendCnt({cnt_temp.oxy{2}, cnt_temp.oxy{4}, cnt_temp.oxy{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); % merged mental arithmetic cnts
    
    % --- (6) band-pass filtering -----------------------------------------
    High_cut = 0.1; % Hz
    ord = 6;
    [high_b, high_a] = butter(ord, High_cut/cnt.ment.deoxy.fs*2, 'low');

    cnt.ment.deoxy = proc_filtfilt(cnt.ment.deoxy, high_b, high_a); 
    cnt.ment.oxy   = proc_filtfilt(cnt.ment.oxy, high_b, high_a);
    
    clear high_b high_a
    
    Low_cut = 0.01; % Hz
    ord = 6;
    [low_b, low_a] = butter(ord, Low_cut/cnt.ment.deoxy.fs*2, 'high');

    cnt.ment.deoxy = proc_filtfilt(cnt.ment.deoxy, low_b, low_a);
    cnt.ment.oxy   = proc_filtfilt(cnt.ment.oxy, low_b, low_a);
    
    clear low_b low_a

    % --- (7) channel selection -------------------------------------------
    FrontalChannel = {'AF7Fp1','AF3Fp1','AF3AFz','FpzFp1','FpzAFz','FpzFp2','AF4AFz','AF4Fp2','AF8Fp2'};
    MotorChannel = {'C5CP5','C5FC5','C5C3','FC3FC5','FC3C3','FC3FC1','CP3CP5','CP3C3','CP3CP1','C1C3','C1FC1','C1CP1','C2FC2','C2CP2','C2C4','FC4FC2','FC4C4','FC4FC6','CP4CP6','CP4CP2','CP4C4','C6CP6','C6C4','C6FC6'};
    OccipitalChannel = {'OzPOz','OzO1','OzO2'};
    AllChannel = [FrontalChannel MotorChannel OccipitalChannel];    

    cnt_org.ment.deoxy = cnt.ment.deoxy;                                   % backup
    cnt_org.ment.oxy   = cnt.ment.oxy;                                     % backup

    cnt.ment.deoxy = proc_selectChannels(cnt.ment.deoxy, FrontalChannel);
    cnt.ment.oxy   = proc_selectChannels(cnt.ment.oxy, FrontalChannel);

    % --- (8) segmentation (epoching) -------------------------------------
    ival_epo  = [-5 20]*1000;                                              % epoch range (unit: msec)

    epo.nirs.ment.deoxy = proc_segmentation(cnt.ment.deoxy, mrk.ment.deoxy, ival_epo);
    epo.nirs.ment.oxy   = proc_segmentation(cnt.ment.oxy, mrk.ment.oxy, ival_epo);

    % --- (9) baseline correction -----------------------------------------
    ival_base = [-5 -2]*1000;                                              % baseline correction range (unit: msec)

    epo.nirs.ment.deoxy = proc_baseline(epo.nirs.ment.deoxy,ival_base);
    epo.nirs.ment.oxy   = proc_baseline(epo.nirs.ment.oxy,ival_base);

    %%%%%% EEG Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- (1) load occular artifact-free eeg data
    subdir_list = {['subject ', num2str(sub_ind(temp_subNum)), '\with occular artifact']}; % subject label
    loadDir = fullfile(EegMyDataDir, subdir_list{1});
    cd(loadDir);
    load cnt; load mrk, load mnt;                                          % load continous eeg signal (cnt), marker (mrk) and montage (mnt)

    clear cnt_temp mrk_temp
    % --- (2) merge cnts in each session ----------------------------------
    cnt_temp = cnt; mrk_temp = mrk;                                        % save data temporarily
    clear cnt mrk;
    [cnt.imag, mrk.imag] = proc_appendCnt({cnt_temp{1}, cnt_temp{3}, ...   % merged motor imagery cnts
        cnt_temp{5}}, {mrk_temp{1}, mrk_temp{3}, mrk_temp{5}}); 
    [cnt.ment, mrk.ment] = proc_appendCnt({cnt_temp{2}, cnt_temp{4}, ...   % merged mental arithmetic cnts
        cnt_temp{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); 
    
    % --- (3) convert rawdata to data without EOG artifacts by ICA --------
    load cnt_ment_afterICA
    cnt.ment.x = double(EEG.data)';
    clear EEG

    % --- (3) Select EEG channels only (excluding EOG channels) -----------
    % clab = {'F7','FAF5','F3','AFp1','AFp2','FAF6','F4','F8','FAF1','FAF2', ...
    %         'Cz','Pz','CFC5','CFC3','CCP5','CCP3','T7','P7','P3','PPO1', ...
    %         'OPO1','OPO2','PPO2','P4','CFC4','CFC6','CCP4','CCP6','P8', ...
    %         'T8','VEOG','HEOG'}
    cnt.ment = proc_selectChannels(cnt.ment,'not','*EOG'); 
    mnt.ment = mnt_setElectrodePositions(cnt.ment.clab);

    % --- (6) segmentation (epoching) -------------------------------------
    ival_epo  = [-5 20]*1000;                                              % epoch range (unit: msec)
    ival_base = [-5 -2]*1000;                                              % baseline correction range (unit: msec)

    epo.eeg.ment = proc_segmentation(cnt.ment, mrk.ment, ival_epo);
    epo.eeg.ment = proc_baseline(epo.eeg.ment,ival_base);

    % --- (7) channel selection for common spatial pattern (CSP) ----------
    % Motor: FCC5h, FCC3h, FCC4h, FCC6h, Cz, CCP5h, CCP3h, CCP4h, CCP6h
    % Parietal: P7, P3, Pz, P4, P8, PPO1h, PPO2h
    % Frontal: AFp1, AFp2, AFF5h, AFF1h, AFF2h, AFF6h, F7, F3, F4, F8
    % Occipital: POO1, POO2
    % Temporal: T7, T8
    MotorChannel = {'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'Cz', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h'};
    ParietalChannel = {'P7', 'P3', 'Pz', 'P4', 'P8'};
    FrontalChannel = {'AFp1', 'AFp2', 'AFF5h', 'AFF1h', 'AFF2h', 'AFF6h', 'F7', 'F3', 'F4', 'F8'};
    OccipitalChannel = {'PPO1h', 'PPO2h', 'POO1', 'POO2'};
    AllChannels = [FrontalChannel ParietalChannel MotorChannel OccipitalChannel];

    cnt_org.ment = cnt.ment;                                               % backup
    cnt.ment = proc_selectChannels(cnt.ment, [FrontalChannel]);
  
    % --- (8) narrow frequency band selection for CSP ---------------------
    band_csp.ment = select_bandnarrow(cnt.ment, mrk.ment, [0 10]*1000);    % band selection using 0~10 sec epoch for mental arithmetic

    Ind_band_csp(temp_subNum, :) = band_csp.ment;
    % --- (9) Cheby2 bandpass filter with a passband of band_csp, with at most Rp dB of passband ripple and at least Rs dB attenuation in the stopbands that are 3 Hz wide on both sides of the passband
    clear ord
    % mental arithmetic
    Wp.ment = band_csp.ment/epo.eeg.ment.fs*2;
    Ws.ment = [band_csp.ment(1)-3, band_csp.ment(end)+3]/epo.eeg.ment.fs*2;
    Rp.ment = 3;                                                           % in dB
    Rs.ment = 30;                                                          % in dB 
    [ord.ment, Ws.ment] = cheb2ord(Wp.ment, Ws.ment, Rp.ment, Rs.ment);
    [filt_b.ment, filt_a.ment] = cheby2(ord.ment, Rs.ment, Ws.ment); 
 
    epo.eeg.ment = proc_filtfilt(epo.eeg.ment, filt_b.ment, filt_a.ment);  

    %% classification by using moving time windows
        
    for foldIdx = 1:6
        
        Test_foldIdx = (1+(foldIdx-1)*10) : (10+(foldIdx-1)*10);
        Train_foldIdx = ones(60, 1);
        Train_foldIdx(Test_foldIdx) = 0;
        Train_foldIdx = find(Train_foldIdx == 1);
        
        % make training and test set
        epo_TR.nirs.ment.deoxy = proc_selectEpochs(epo.nirs.ment.deoxy, Train_foldIdx);
        epo_TR.nirs.ment.oxy = proc_selectEpochs(epo.nirs.ment.oxy, Train_foldIdx);
        epo_TR.eeg.ment = proc_selectEpochs(epo.eeg.ment, Train_foldIdx);

        epo_TE.nirs.ment.deoxy = proc_selectEpochs(epo.nirs.ment.deoxy, Test_foldIdx);
        epo_TE.nirs.ment.oxy = proc_selectEpochs(epo.nirs.ment.oxy, Test_foldIdx);
        epo_TE.eeg.ment = proc_selectEpochs(epo.eeg.ment, Test_foldIdx);

        % --- Extract features in training set ----------------------------
        ival_TR = [0*1000, 10*1000];

        % average
        ave_TR.ment.deoxy = proc_meanAcrossTime(epo_TR.nirs.ment.deoxy, ival_TR);
        ave_TR.ment.oxy   = proc_meanAcrossTime(epo_TR.nirs.ment.oxy,   ival_TR);

        % slope
        slope_TR.ment.deoxy = proc_slopeAcrossTime(epo_TR.nirs.ment.deoxy, ival_TR);
        slope_TR.ment.oxy   = proc_slopeAcrossTime(epo_TR.nirs.ment.oxy,   ival_TR);

        % segment for log-variance
        ival_TR = [0*1000, 10*1000];
        segment_TR.ment = proc_selectIval(epo_TR.eeg.ment, ival_TR);

        % --- Extract features in test dataset ----------------------------
        StepSize = 0.5*1000;                                               % msec
        WindowSize = 1*1000;                                               % msec
        ival_start = (0*1000:StepSize:20*1000-WindowSize)';
        ival_end = ival_start+WindowSize;
        ival_TE = [ival_start, ival_end];
        nStep = length(ival_TE);
        clear StepSize WindowSize ival_start ival_end

        % average
        for stepIdx = 1:nStep
            ave_TE.ment.deoxy{stepIdx} = proc_meanAcrossTime(epo_TE.nirs.ment.deoxy, ival_TE(stepIdx,:));
            ave_TE.ment.oxy{stepIdx}   = proc_meanAcrossTime(epo_TE.nirs.ment.oxy,   ival_TE(stepIdx,:));
        end

        % slope
        for stepIdx = 1:nStep
            slope_TE.ment.deoxy{stepIdx} = proc_slopeAcrossTime(epo_TE.nirs.ment.deoxy, ival_TE(stepIdx,:));
            slope_TE.ment.oxy{stepIdx}   = proc_slopeAcrossTime(epo_TE.nirs.ment.oxy,   ival_TE(stepIdx,:));
        end

        % segment for log-variance
        for stepIdx = 1:nStep
            segment_TE.ment{stepIdx} = proc_selectIval(epo_TE.eeg.ment, ival_TE(stepIdx,:));
        end

        group_TR.ment = epo_TR.nirs.ment.deoxy.y;                          % epo.ment.deoxy.y == epo.ment.oxy.y
        group_TE.ment = epo_TE.nirs.ment.deoxy.y;

        % --- mental arithmetic -------------------------------------------
        %%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % HbR
        x_train.deoxy.x    = [squeeze(ave_TR.ment.deoxy.x(:,:,:)); squeeze(slope_TR.ment.deoxy.x(:,:,:))];
        x_train.deoxy.y    = squeeze(ave_TR.ment.deoxy.y(:,:));
        x_train.deoxy.clab = ave_TR.ment.deoxy.clab;
        % HbO
        x_train.oxy.x      = [squeeze(ave_TR.ment.oxy.x(:,:,:)); squeeze(slope_TR.ment.oxy.x(:,:,:))];
        x_train.oxy.y      = squeeze(ave_TR.ment.oxy.y(:,:));
        x_train.oxy.clab   = ave_TR.ment.oxy.clab;
        % eeg
        x_train.eeg.x    = segment_TR.ment.x(:,:,:);
        x_train.eeg.y    = segment_TR.ment.y(:,:);
        x_train.eeg.clab = segment_TR.ment.clab;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%% CSP for EEG %%%%%%%%%%%%%%%%%%%%%%%%%%
        [csp_train, CSP_W, CSP_EIG, CSP_A] = proc_cspAuto(x_train.eeg);
        csp_train.x = csp_train.x(:,[1 2 end-1 end],:);

        csp_train.y    = x_train.eeg.y;
        csp_train.clab = x_train.eeg.clab;

        % variance and logarithm 
        var_train = proc_variance(csp_train);
        logvar_train = proc_logarithm(var_train);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % feature vector
        fv_train.deoxy.x = x_train.deoxy.x; fv_train.deoxy.y = x_train.deoxy.y; fv_train.deoxy.className = {'MA','BL'};
        fv_train.oxy.x   = x_train.oxy.x;   fv_train.oxy.y   = x_train.oxy.y;   fv_train.oxy.className   = {'MA','BL'};
        fv_train.eeg.x   = logvar_train.x;  fv_train.eeg.y   = x_train.eeg.y;   fv_train.eeg.className   = {'MA','BL'};

        % for eeg only
        fv_train.eeg.x = squeeze(fv_train.eeg.x);
        y_train  = group_TR.ment(:,:);

        % train classifier 
        C.deoxy = train_RLDAshrink(fv_train.deoxy.x, y_train);
        C.oxy   = train_RLDAshrink(fv_train.oxy.x  , y_train);
        C.eeg   = train_RLDAshrink(fv_train.eeg.x  , y_train);

        %%%%%%%%%%%%%%%%%%%%%% train meta-classifier %%%%%%%%%%%%%%%%%%%%%%
        map_train.deoxy.x = apply_separatingHyperplane(C.deoxy, fv_train.deoxy.x);
        map_train.oxy.x = apply_separatingHyperplane(C.oxy, fv_train.oxy.x);
        map_train.eeg.x = apply_separatingHyperplane(C.eeg, fv_train.eeg.x);

        % meta1: HbR+HbO / meta2: HbR+EEG / meta3: HbO+EEG / meta4: HbR+HbO+EEG
        fv_train.meta1.x = [map_train.deoxy.x; map_train.oxy.x]; 
        fv_train.meta2.x = [map_train.deoxy.x; map_train.eeg.x];
        fv_train.meta3.x = [map_train.oxy.x; map_train.eeg.x];
        fv_train.meta4.x = [map_train.deoxy.x;  map_train.oxy.x; map_train.eeg.x];

        y_map_train = y_train;

        % train classifier
        C.meta1 = train_RLDAshrink(fv_train.meta1.x, y_map_train);
        C.meta2 = train_RLDAshrink(fv_train.meta2.x, y_map_train);
        C.meta3 = train_RLDAshrink(fv_train.meta3.x, y_map_train);
        C.meta4 = train_RLDAshrink(fv_train.meta4.x, y_map_train);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for stepIdx = 1:size(ival_TE, 1) 
            for temp_trialNum = 1:size(epo_TE.nirs.ment.deoxy.x, 3)
                % Test set 
                % HbR
                x_test.deoxy.x    = [squeeze(ave_TE.ment.deoxy{stepIdx}.x(:,:,temp_trialNum)) squeeze(slope_TE.ment.deoxy{stepIdx}.x(:,:,temp_trialNum))];
                x_test.deoxy.y    = squeeze(ave_TE.ment.deoxy{stepIdx}.y(:,temp_trialNum));
                x_test.deoxy.clab = ave_TE.ment.deoxy{stepIdx}.clab;
                % HbO
                x_test.oxy.x    = [squeeze(ave_TE.ment.oxy{stepIdx}.x(:,:,temp_trialNum)) squeeze(slope_TE.ment.oxy{stepIdx}.x(:,:,temp_trialNum))];
                x_test.oxy.y    = squeeze(ave_TE.ment.oxy{stepIdx}.y(:,temp_trialNum));
                x_test.oxy.clab = ave_TE.ment.oxy{stepIdx}.clab;
                % EEG
                x_test.eeg.x    = segment_TE.ment{stepIdx}.x(:,:,temp_trialNum);
                x_test.eeg.y    = segment_TE.ment{stepIdx}.y(:,temp_trialNum);
                x_test.eeg.clab = segment_TE.ment{stepIdx}.clab;

                for testIdx = 1 : size(x_test.eeg.x, 3)
                    csp_test.x(:,:,testIdx) = x_test.eeg.x(:,:,testIdx)*CSP_W;
                end

                csp_test.x = csp_test.x(:,[1 2 end-1 end],:);

                csp_test.y     = x_test.eeg.y;
                csp_test.clab  = x_test.eeg.clab;

                var_test  = proc_variance(csp_test);
                logvar_test  = proc_logarithm(var_test);

                % feature vector
                fv_test.deoxy.x = x_test.deoxy.x'; fv_test.deoxy.y = x_test.deoxy.y; fv_test.deoxy.className = {'MA','BL'};
                fv_test.oxy.x   = x_test.oxy.x';   fv_test.oxy.y   = x_test.oxy.y;   fv_test.oxy.className   = {'MA','BL'};
                fv_test.eeg.x   = logvar_test.x';  fv_test.eeg.y   = x_test.eeg.y;   fv_test.eeg.className   = {'MA','BL'};

                % for eeg only 
                fv_test.eeg.x  = squeeze(fv_test.eeg.x);
                y_test   = vec2ind(group_TE.ment(:,temp_trialNum));        % 1 = MA, 2 = BL

                map_test.deoxy.x = apply_separatingHyperplane(C.deoxy, fv_test.deoxy.x);
                map_test.oxy.x = apply_separatingHyperplane(C.oxy, fv_test.oxy.x);
                map_test.eeg.x = apply_separatingHyperplane(C.eeg, fv_test.eeg.x);

                fv_test.meta1.x  = [map_test.deoxy.x; map_test.oxy.x];
                fv_test.meta2.x  = [map_test.deoxy.x; map_test.eeg.x];
                fv_test.meta3.x  = [map_test.oxy.x; map_test.eeg.x];
                fv_test.meta4.x  = [map_test.deoxy.x; map_test.oxy.x; map_test.eeg.x];

                % classification  
                grouphat{temp_subNum}{foldIdx}.deoxy(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.deoxy, fv_test.deoxy.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.deoxy(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.deoxy(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.deoxy(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.deoxy(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.oxy(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.oxy, fv_test.oxy.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.oxy(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.oxy(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.oxy(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.oxy(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.eeg(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.eeg, fv_test.eeg.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.eeg(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.eeg(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.eeg(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.eeg(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.meta1(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.meta1, fv_test.meta1.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.meta1(temp_trialNum, stepIdx) < 0  
                    grouphat{temp_subNum}{foldIdx}.meta1(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.meta1(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.meta1(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.meta2(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.meta2, fv_test.meta2.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.meta2(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.meta2(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.meta2(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.meta2(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.meta3(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.meta3, fv_test.meta3.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.meta3(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.meta3(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.meta3(temp_trialNum, stepIdx) > 0
                    grouphat{temp_subNum}{foldIdx}.meta3(temp_trialNum, stepIdx) = 0;
                end

                grouphat{temp_subNum}{foldIdx}.meta4(temp_trialNum, stepIdx) = apply_separatingHyperplane(C.meta4, fv_test.meta4.x);      % use custom function
                if grouphat{temp_subNum}{foldIdx}.meta4(temp_trialNum, stepIdx) < 0 
                    grouphat{temp_subNum}{foldIdx}.meta4(temp_trialNum, stepIdx) = 1;
                elseif grouphat{temp_subNum}{foldIdx}.meta4(temp_trialNum, stepIdx) > 0 
                    grouphat{temp_subNum}{foldIdx}.meta4(temp_trialNum, stepIdx) = 0;
                end
                
                if y_test == 1
                    grouphat{temp_subNum}{foldIdx}.y_test(temp_trialNum, :) = 1; % 0 = MA, 1 = BL
                elseif y_test == 2
                    grouphat{temp_subNum}{foldIdx}.y_test(temp_trialNum, :) = 0; % 0 = MA, 1 = BL
                end
                grouphat{temp_subNum}{foldIdx}.ClassLabel = {'MA', 'BL'};

                clear cmat csp_test 
            end
             
            cmat.deoxy(:,:,foldIdx)  = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.deoxy(:, stepIdx));
            cmat.oxy(:,:,foldIdx)    = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.oxy(:, stepIdx));
            cmat.eeg(:,:,foldIdx)    = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.eeg(:, stepIdx));
            cmat.meta1(:,:,foldIdx)  = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.meta1(:, stepIdx));
            cmat.meta2(:,:,foldIdx)  = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.meta2(:, stepIdx));
            cmat.meta3(:,:,foldIdx)  = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.meta3(:, stepIdx));
            cmat.meta4(:,:,foldIdx)  = confusionmat(grouphat{temp_subNum}{foldIdx}.y_test(:, :), grouphat{temp_subNum}{foldIdx}.meta4(:, stepIdx));
            
            acc.deoxy(foldIdx, stepIdx)    = trace((sum(cmat.deoxy,3))) / sum(sum(sum(cmat.deoxy,3),2),1);
            acc.oxy(foldIdx, stepIdx)      = trace((sum(cmat.oxy,3)))   / sum(sum(sum(cmat.oxy,3),2),1);
            acc.eeg(foldIdx, stepIdx)      = trace((sum(cmat.eeg,3)))   / sum(sum(sum(cmat.eeg,3),2),1);
            acc.meta1(foldIdx, stepIdx)    = trace((sum(cmat.meta1,3)))  / sum(sum(sum(cmat.meta1,3),2),1);
            acc.meta2(foldIdx, stepIdx)    = trace((sum(cmat.meta2,3)))  / sum(sum(sum(cmat.meta2,3),2),1);
            acc.meta3(foldIdx, stepIdx)    = trace((sum(cmat.meta3,3)))  / sum(sum(sum(cmat.meta3,3),2),1);
            acc.meta4(foldIdx, stepIdx)    = trace((sum(cmat.meta4,3)))  / sum(sum(sum(cmat.meta4,3),2),1);
            
        end
        
        clear C y_map_test y_map_train fv_test fv_train logvar_test logvar_train var_test var_train csp_test csp_train CSP_W CSP_EIG CSP_A x_test x_train cmat
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    mean_acc.deoxy(temp_subNum, :)  = mean(acc.deoxy,1);
    mean_acc.oxy(temp_subNum, :)    = mean(acc.oxy,1);
    mean_acc.eeg(temp_subNum, :)    = mean(acc.eeg,1);
    mean_acc.meta1(temp_subNum, :)  = mean(acc.meta1,1);
    mean_acc.meta2(temp_subNum, :)  = mean(acc.meta2,1);
    mean_acc.meta3(temp_subNum, :)  = mean(acc.meta3,1);
    mean_acc.meta4(temp_subNum, :)  = mean(acc.meta4,1);
    
    waitbar(temp_subNum/size(sub_ind, 2))

end

%% Calculate grand-averaged classificaiton accuracies 
% (you don't need to run this part if you dont want to check grand-averaged classification accuracies for each modality)

% for temp_subNum = 1:size(sub_ind, 2)
%     
%     for temp_foldIdx = 1:size(grouphat{temp_subNum}, 2)
%         
%         MA_Labels = find(grouphat{temp_subNum}{temp_foldIdx}.y_test == 1);
%         BL_Labels = find(grouphat{temp_subNum}{temp_foldIdx}.y_test == 0);
% 
%         tp_MA_deoxy(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.deoxy(MA_Labels, :), 1);
%         tp_BL_deoxy(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.deoxy(BL_Labels, :), 1);
%         tp_MA_oxy(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.oxy(MA_Labels, :), 1);
%         tp_BL_oxy(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.oxy(BL_Labels, :), 1);
%         tp_MA_eeg(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.eeg(MA_Labels, :), 1);
%         tp_BL_eeg(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.eeg(BL_Labels, :), 1);
%         tp_MA_meta1(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta1(MA_Labels, :), 1);
%         tp_BL_meta1(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta1(BL_Labels, :), 1);
%         tp_MA_meta2(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta2(MA_Labels, :), 1);
%         tp_BL_meta2(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta2(BL_Labels, :), 1);
%         tp_MA_meta3(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta3(MA_Labels, :), 1);
%         tp_BL_meta3(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta3(BL_Labels, :), 1);
%         tp_MA_meta4(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta4(MA_Labels, :), 1);
%         tp_BL_meta4(:, temp_foldIdx) = mean(grouphat{temp_subNum}{temp_foldIdx}.meta4(BL_Labels, :), 1);
% 
%         clear MA_Labels BL_Labels
%     end
%     
%     GA_grouphat.MA.deoxy(temp_subNum, :) = mean(tp_MA_deoxy, 2);
%     GA_grouphat.MA.oxy(temp_subNum, :) = mean(tp_MA_oxy, 2);
%     GA_grouphat.MA.eeg(temp_subNum, :) = mean(tp_MA_eeg, 2);
%     GA_grouphat.MA.meta1(temp_subNum, :) = mean(tp_MA_meta1, 2);
%     GA_grouphat.MA.meta2(temp_subNum, :) = mean(tp_MA_meta2, 2);
%     GA_grouphat.MA.meta3(temp_subNum, :) = mean(tp_MA_meta3, 2);
%     GA_grouphat.MA.meta4(temp_subNum, :) = mean(tp_MA_meta4, 2);
%     
%     GA_grouphat.BL.deoxy(temp_subNum, :) = mean(tp_BL_deoxy, 2);
%     GA_grouphat.BL.oxy(temp_subNum, :) = mean(tp_BL_oxy, 2);
%     GA_grouphat.BL.eeg(temp_subNum, :) = mean(tp_BL_eeg, 2);
%     GA_grouphat.BL.meta1(temp_subNum, :) = mean(tp_BL_meta1, 2);
%     GA_grouphat.BL.meta2(temp_subNum, :) = mean(tp_BL_meta2, 2);
%     GA_grouphat.BL.meta3(temp_subNum, :) = mean(tp_BL_meta3, 2);
%     GA_grouphat.BL.meta4(temp_subNum, :) = mean(tp_BL_meta4, 2);
%     
%     clear tp_MA_deoxy tp_BL_deoxy tp_MA_oxy tp_BL_oxy tp_MA_eeg tp_BL_eeg tp_MA_meta1 tp_BL_meta1 tp_MA_meta2 tp_BL_meta2 tp_MA_meta3 tp_BL_meta3 tp_MA_meta4 tp_BL_meta4
% end
% 
% GrandAveragedGrouphat.MA.deoxy = mean(GA_grouphat.MA.deoxy, 1);
% GrandAveragedGrouphat.MA.oxy = mean(GA_grouphat.MA.oxy, 1);
% GrandAveragedGrouphat.MA.eeg = mean(GA_grouphat.MA.eeg, 1);
% GrandAveragedGrouphat.MA.meta1 = mean(GA_grouphat.MA.meta1, 1);
% GrandAveragedGrouphat.MA.meta2 = mean(GA_grouphat.MA.meta2, 1);
% GrandAveragedGrouphat.MA.meta3 = mean(GA_grouphat.MA.meta3, 1);
% GrandAveragedGrouphat.MA.meta4 = mean(GA_grouphat.MA.meta4, 1);
% 
% GrandAveragedGrouphat.BL.deoxy = mean(GA_grouphat.BL.deoxy, 1);
% GrandAveragedGrouphat.BL.oxy = mean(GA_grouphat.BL.oxy, 1);
% GrandAveragedGrouphat.BL.eeg = mean(GA_grouphat.BL.eeg, 1);
% GrandAveragedGrouphat.BL.meta1 = mean(GA_grouphat.BL.meta1, 1);
% GrandAveragedGrouphat.BL.meta2 = mean(GA_grouphat.BL.meta2, 1);
% GrandAveragedGrouphat.BL.meta3 = mean(GA_grouphat.BL.meta3, 1);
% GrandAveragedGrouphat.BL.meta4 = mean(GA_grouphat.BL.meta4, 1);
% 
% % --- Plot classification outputs for each condition ----------------------
% figure(1)
% set(gcf, 'color', [1 1 1])
% 
% subplot(1,2,1);
% time = (ival_TE(:,2)/1000)-1';
% shadedErrorBar(time, GA_grouphat.MA.meta1, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-+r','markerfacecolor','r', 'linewidth', 2}, 'transparent',1); hold on
% shadedErrorBar(time, GA_grouphat.MA.eeg, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-ob','markerfacecolor','b', 'linewidth', 2}, 'transparent',1); hold on
% shadedErrorBar(time, GA_grouphat.MA.meta4, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-xk','markerfacecolor','k', 'linewidth', 2}, 'transparent',1); hold on
% h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'southwest');
% ylim([0 1]) 
% xlim([1 19])
% title('Control state')
% ylabel('Class labels')
% xlabel('Time (sec)')
% set(gca, 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 15, 'FontWeight', 'bold', 'box', 'on') 
% set(h, 'FontSize', 13)
%  
% subplot(1,2,2);
% shadedErrorBar(time, GA_grouphat.BL.meta1, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-+r','markerfacecolor','r', 'linewidth', 2}, 'transparent',1); hold on
% shadedErrorBar(time, GA_grouphat.BL.eeg, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-ob','markerfacecolor','b', 'linewidth', 2}, 'transparent',1); hold on
% shadedErrorBar(time, GA_grouphat.BL.meta4, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-xk','markerfacecolor','k', 'linewidth', 2}, 'transparent',1); hold on
% h = legend('NIRS', 'EEG', 'Hybrid');
% ylim([0 1])
% xlim([1 19])
% title('Idle state')
% ylabel('Class labels') 
% xlabel('Time (sec)')
% set(gca, 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 15, 'FontWeight', 'bold', 'box', 'on')
% set(h, 'FontSize', 13)
%  
% % --- plot grand-averaged classification accuracies for each modality ----- 
% time = (ival_TE(:,2)/1000)-1';
% figure(2) 
% set(gcf, 'color', [1 1 1]) 
% hold on
% shadedErrorBar(time, mean_acc.meta1, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-+r','markerfacecolor','r', 'linewidth', 2}, 'transparent',1);
% shadedErrorBar(time, mean_acc.eeg, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-ob','markerfacecolor','b', 'linewidth', 2}, 'transparent',1);
% shadedErrorBar(time, mean_acc.meta4, {@mean,@(x) std(x)/sqrt(length(x))},'lineprops',{'-xk','markerfacecolor','k', 'linewidth', 2}, 'transparent',1);
% % h = legend('NIRS', 'EEG', 'Hybrid' ,'location', 'SouthWest');
% ylim([0.45 0.85]) 
% xlim([1 19])
% ylabel('Classification accuracy (%)')
% xlabel('Time (sec)') 
% set(gcf, 'Color', [1 1 1]) 
% set(gca, 'LineWidth', 2, 'FontName', 'Arial', 'FontSize', 15, 'FontWeight', 'bold', 'box', 'on')
  
%% Detect onsets using a template matching algorithm
clc  
clearvars -except grouphat ival_TE jj    

cnt = 1; 

for jj = 1:20                                                              % jj = template size

    for temp_subNum = 1:size(grouphat, 2)
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            
            Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.oxy = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.eeg = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta1 = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta2 = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta3 = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta4 = zeros(size(grouphat{1}{1}.deoxy, 1), 1);
            
            for temp_trialNum = 1:size(grouphat{1}{1}.deoxy, 1)
                tp = ones(1, jj); 
                template = 1*tp;
                % deoxy
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.deoxy(:, 1:20), 2) - size(template, 2) + 1)
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.deoxy(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % oxy
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.oxy(:, 1:20), 2) - size(template, 2) + 1)
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.oxy(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % eeg
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.eeg(:, 1:20), 2) - size(template, 2) + 1) 
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.eeg(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % meta1
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.meta1(:, 1:20), 2) - size(template, 2) + 1) 
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.meta1(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % meta2
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.meta2(:, 1:20), 2) - size(template, 2) + 1) 
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.meta2(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % meta3
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.meta3(:, 1:20), 2) - size(template, 2) + 1) 
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.meta3(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
                % meta4
                for ii = 1:(size(grouphat{temp_subNum}{temp_foldIdx}.meta4(:, 1:20), 2) - size(template, 2) + 1) 
                    if (template - grouphat{temp_subNum}{temp_foldIdx}.meta4(temp_trialNum, 1+(ii-1):size(template, 2)+(ii-1))) == 0
                        Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(temp_trialNum, 1) = size(template, 2)+(ii-1);
                        break
                    end
                end
            end
            Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
            Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(:, 2) = grouphat{temp_subNum}{temp_foldIdx}.y_test;
        
        end
    end

    % --- Calculate TR and FR ---------------------------------------------
    Final_Result{cnt}.TPR = zeros(size(grouphat, 2), 7);
    Final_Result{cnt}.TNR = zeros(size(grouphat, 2), 7);
    Final_Result{cnt}.FPR = zeros(size(grouphat, 2), 7);
    Final_Result{cnt}.FNR = zeros(size(grouphat, 2), 7);
    
    for temp_subNum = 1:size(grouphat, 2)

        %%% deoxy
        for temp_foldIdx = 1:size(grouphat{1,1}, 2) 
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end 
        Final_Result{cnt}.TPR(temp_subNum, 1) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 1) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 1) = mean(sum(FP)/size(FR_Idx, 1)*100); 
        Final_Result{cnt}.FP_min(temp_subNum, 1) = mean((sum(FP)/5)*3); 
        Final_Result{cnt}.FNR(temp_subNum, 1) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% oxy
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 2) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 2) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 2) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 2) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 2) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% EEG
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 3) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 3) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 3) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 3) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 3) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% meta1
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 4) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 4) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 4) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 4) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 4) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% meta2
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 5) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 5) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 5) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 5) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 5) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% meta3
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 6) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 6) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 6) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 6) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 6) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

        %%% meta4
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            % True positive rate / true negative rate
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(TR_Idx(ii), 1) ~= 0
                    TP(ii, temp_foldIdx) = 1;
                    TN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(TR_Idx(ii), 1) == 0
                    TP(ii, temp_foldIdx) = 0;
                    TN(ii, temp_foldIdx) = 1;
                end
            end
            % False positive rate / false negative rate
            FR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(:, 2) == 0);
            for ii = 1:size(FR_Idx, 1)
                if Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(FR_Idx(ii), 1) ~= 0
                    FP(ii, temp_foldIdx) = 1;
                    FN(ii, temp_foldIdx) = 0;
                elseif Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(FR_Idx(ii), 1) == 0
                    FP(ii, temp_foldIdx) = 0;
                    FN(ii, temp_foldIdx) = 1;
                end
            end
        end
        Final_Result{cnt}.TPR(temp_subNum, 7) = mean(sum(TP)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.TNR(temp_subNum, 7) = mean(sum(TN)/size(TR_Idx, 1)*100);
        Final_Result{cnt}.FPR(temp_subNum, 7) = mean(sum(FP)/size(FR_Idx, 1)*100);
        Final_Result{cnt}.FP_min(temp_subNum, 7) = mean(sum(FP)/5*3);
        Final_Result{cnt}.FNR(temp_subNum, 7) = mean(sum(FN)/size(FR_Idx, 1)*100);
        clear TR_Idx FR_Idx TP TN FP FN

    end 

    clearvars -except Final_Result Onset_Detection grouphat ival_TE jj GA_TPR GA_FPR GA_TNR GA_FNR GA_FP_min GA_OnsetTime cnt

    % --- Calculate Onset detection time ----------------------------------
    Final_Result{cnt}.OnsetDetectionTime = zeros(size(grouphat, 2), 7);

    for temp_subNum = 1:size(grouphat, 2)
        
        %%% deoxy
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.deoxy(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 1) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% oxy
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.oxy(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 2) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% eeg
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.eeg(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 3) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% meta1
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.meta1(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 4) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% meta2
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.meta2(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 5) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% meta3
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.meta3(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 6) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
        %%% meta4
        for temp_foldIdx = 1:size(grouphat{1,1}, 2)
            TR_Idx = find(Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(:, 2) == 1);
            for ii = 1:size(TR_Idx, 1)
                TimeIdx = Onset_Detection{temp_subNum}{temp_foldIdx}.meta4(TR_Idx(ii), 1);
                if TimeIdx ~= 0
                    tp(ii, temp_foldIdx) = ival_TE(TimeIdx, 2)/1000;
                elseif TimeIdx == 0
                    tp(ii, temp_foldIdx) = NaN;
                end
            end 
            tp_OnsetDetectionTime(:, temp_foldIdx) = nanmean(tp(:, temp_foldIdx));
        end
        Final_Result{cnt}.OnsetDetectionTime(temp_subNum, 7) = nanmean(tp_OnsetDetectionTime);
        clear tp_OnsetDetectionTime tp
        
    end

    clearvars -except Final_Result Onset_Detection grouphat ival_TE jj GA_TPR GA_FPR GA_TNR GA_FNR GA_FP_min GA_OnsetTime cnt

    GA_TPR(cnt, :) = mean(Final_Result{cnt}.TPR(:, :), 1);
    GA_FPR(cnt, :) = mean(Final_Result{cnt}.FPR(:, :), 1);
    GA_FP_min(cnt, :) = mean(Final_Result{cnt}.FP_min(:, :), 1);
    GA_TNR(cnt, :) = mean(Final_Result{cnt}.TNR(:, :), 1);
    GA_FNR(cnt, :) = mean(Final_Result{cnt}.FNR(:, :), 1);
    GA_OnsetTime(cnt, :) = nanmean(Final_Result{cnt}.OnsetDetectionTime(:, [4 3 7]), 1);
    
    cnt = cnt + 1;
    
end 

% --- Calculate Individual AUC --------------------------------------------
for temp_subNum = 1:size(Final_Result{1, 1}.TPR, 1)
    for ii = 1:size(Final_Result, 2)
        Ind_TPR{temp_subNum}(ii, :) = Final_Result{1, ii}.TPR(temp_subNum, :);
        Ind_FPR{temp_subNum}(ii, :) = Final_Result{1, ii}.FPR(temp_subNum, :);
    end
    Ind_TPR{temp_subNum} = [ones(1, 7)*100; Ind_TPR{temp_subNum}; zeros(1, 7);];
    Ind_FPR{temp_subNum} = [ones(1, 7)*100; Ind_FPR{temp_subNum}; zeros(1, 7);];
    
    Ind_AUC(temp_subNum, 1) = trapz(flipud(Ind_FPR{temp_subNum}(:, 4)), flipud(Ind_TPR{temp_subNum}(:, 4)));
    Ind_AUC(temp_subNum, 2) = trapz(flipud(Ind_FPR{temp_subNum}(:, 3)), flipud(Ind_TPR{temp_subNum}(:, 3)));
    Ind_AUC(temp_subNum, 3) = trapz(flipud(Ind_FPR{temp_subNum}(:, 7)), flipud(Ind_TPR{temp_subNum}(:, 7)));
end

% Plotting 
figure()
az = boxplot(Ind_AUC);
set(az, {'linew'}, {2})
set(gcf, 'Color', [1 1 1])
set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'YTickLabel', {}, 'XTickLabel', {'NIRS', 'EEG', 'Hybrid'}, 'YTick', {})
ylabel('Individual AUC')
ylim([3200 13000])

% Statistical analysis 
% normalitytest(Ind_AUC(:, 1)') % Shapiro-Francia Test 
% [p_AUC] = friedman(Ind_AUC);
% if p_AUC <= 0.05
%     p_AUC_hoc(1) = signrank(Ind_AUC(:, 1), Ind_AUC(:, 2));
%     p_AUC_hoc(2) = signrank(Ind_AUC(:, 1), Ind_AUC(:, 3));
%     p_AUC_hoc(3) = signrank(Ind_AUC(:, 2), Ind_AUC(:, 3));
% end
% [corrected_p_AUC, h_AUC] = bonf_holm(p_AUC_hoc, 0.05);

% --- Calculate GA AUC ----------------------------------------------------
GA_TPR = [ones(1, 7)*100; GA_TPR; zeros(1, 7);];
GA_FPR = [ones(1, 7)*100; GA_FPR; zeros(1, 7);];
GA_TNR = [zeros(1, 7); GA_TNR; ones(1, 7)*100;];
GA_FNR = [zeros(1, 7); GA_FNR; ones(1, 7)*100;]; 

AUC(1) = trapz(flipud(GA_FPR(:, 4)), flipud(GA_TPR(:, 4)));
AUC(2) = trapz(flipud(GA_FPR(:, 3)), flipud(GA_TPR(:, 3)));
AUC(3) = trapz(flipud(GA_FPR(:, 7)), flipud(GA_TPR(:, 7)));
% -------------------------------------------------------------------------


% cd('D:\Works\2018\Researches\2_Hybrid Onset Detection\Datasets')
% save('Final_Result_190301.mat', 'Final_Result') 

% --- Plot ROC curces for each modality -----------------------------------
figure
plot(GA_FPR(:, 4), GA_TPR(:, 4), '-+r', 'linewidth', 2); hold on
plot(GA_FPR(:, 3), GA_TPR(:, 3), '-ob', 'linewidth', 2); hold on
plot(GA_FPR(:, 7), GA_TPR(:, 7), '-xk', 'linewidth', 2); hold on
xlim([0 100])
ylim([0 100])
h = legend('NIRS', 'EEG', 'Hybrid', 'HbR+HbO', 'HbR+EEG', 'HbO+EEG', 'Hybrid', 'location', 'SouthEast');
set(gcf, 'Color', [1 1 1])
set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'Ytick', '', 'XTick', '')
ylabel('True Positive Rate (%)')
xlabel('False Positivie Rate (%)')

% --- Plot onset detection times for each modality ------------------------
figure
plot(GA_OnsetTime(:, 1), '-+r', 'linewidth', 2); hold on
plot(GA_OnsetTime(:, 2), '-ob', 'linewidth', 2); hold on 
plot(GA_OnsetTime(:, 3), '-xk', 'linewidth', 2); hold on
xlim([1 size(GA_OnsetTime, 1)])
h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'SouthEast');
set(gcf, 'Color', [1 1 1])
set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold')
ylabel('Onset detection time (sec)')
xlabel('The size of template')

% --- plot TPRs according to different template sizes ---------------------
figure
subplot(1,2,1); 
plot(GA_TPR(2:10, 4), '-+r', 'linewidth', 2); hold on
plot(GA_TPR(2:10, 3), '-ob', 'linewidth', 2); hold on
plot(GA_TPR(2:10, 7), '-xk', 'linewidth', 2); hold on
xlim([1 9])
h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'SouthWest');
set(gcf, 'Color', [1 1 1])
set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold')
ylabel('True positive rate (%)')
xlabel('Template size')

% --- plot FPRs according to different template sizes ---------------------
subplot(1,2,2);
plot(GA_FP_min(1:9, 4), '-+r', 'linewidth', 2); hold on
plot(GA_FP_min(1:9, 3), '-ob', 'linewidth', 2); hold on
plot(GA_FP_min(1:9, 7), '-xk', 'linewidth', 2); hold on
xlim([1 9])
% h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'SouthEast');
set(gcf, 'Color', [1 1 1])
set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold')
ylabel('False positive rate (FP/min)')
xlabel('Template size')

% cd('D:\Works\2018\Researches\2_Hybrid Onset Detection\Results\Pseudo online test\Trials_50')
% save('grouphat.mat', 'grouphat')
% save('ival_TE.mat', 'ival_TE')
% save('Final_Result.mat', 'Final_Result')
% save('Onset_Detection.mat', 'Onset_Detection')

% subplot(2,2,3); 
% plot(GA_FPR(2:10, 4), '-+r', 'linewidth', 2); hold on
% plot(GA_FPR(2:10, 3), '-ob', 'linewidth', 2); hold on
% plot(GA_FPR(2:10, 7), '-xk', 'linewidth', 2); hold on
% xlim([1 9])
% % h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'NorthEast');
% set(gcf, 'Color', [1 1 1])
% set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'XTickLabel', '', 'YTickLabel', '', 'XTick', '', 'YTick', '')
% ylabel('FPR (%)')
% xlabel('Template size')
% 
% subplot(2,2,4);
% plot(GA_FNR(2:10, 4), '-+r', 'linewidth', 2); hold on
% plot(GA_FNR(2:10, 3), '-ob', 'linewidth', 2); hold on
% plot(GA_FNR(2:10, 7), '-xk', 'linewidth', 2); hold on
% xlim([1 9])
% % h = legend('NIRS', 'EEG', 'Hybrid', 'location', 'SouthEast');
% set(gcf, 'Color', [1 1 1])
% set(gca, 'box', 'on', 'linewidth', 2, 'FontName', 'Arial', 'FontSize', 18, 'FontWeight', 'bold', 'XTickLabel', '', 'YTickLabel', '', 'XTick', '', 'YTick', '')
% ylabel('FNR (%)')
% xlabel('Template size')