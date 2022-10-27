% REST_0.m
%
% covert raw file to mat file for ICA
%
% created: 2022.10.11
% author: Gi-Hwan Shin
%% init
clc; clear; close all;
%% path setting
temp = pwd;
list = split(temp,'\');

path = [];
for i=1:length(list)-2
    path = [path,list{i},'\'];
end
%% addpath
addpath([path,'Wake_Sleep\Lib\BCILAB-master\']);
addpath([path,'Wake_Sleep\Lib\eeglab14_1_2b\']);
eeglab;
%% preprocessing
% 1. resampling
% 2. band pass filtering
% 3. removal of EOG channels or channel rejection
% 4. average referencing
% 5. laplacian filter
% 6. epoching (Total NAP)

fs = 250;
band = [0.5 100];
Trigger = {'S 91', 'S 92', 'S 93'};

removal = [9,11,21,24,26,33,35]; % non resting-state
DirGroup = dir([path,'Wake_Sleep\Data\BN\*.vhdr']);
for n=37
    if sum(n==removal)==1
        continue;
    end

    NAME = DirGroup(n).name;
    [EEG,c1]=pop_loadbv([path,'Wake_Sleep\Data\BN\'], NAME, [], []); % EEG 
   
    %% 1. resampling
    EEG=pop_resample(EEG, fs); 
    
    %% 2. band pass filtering
    EEG=pop_eegfiltnew(EEG, band(1), band(2)); 

    %% 3.channel select
    EOG = EEG.data(29:32,:);
    EEG=pop_chanedit(EEG, 'lookup','standard-10-5-cap385.elp');
    EEG=pop_select(EEG,'nochannel',{'REOG' 'LHEOG' 'UVEOG' 'LVEOG'});
    
    %% 4. EEG & EOG epoching (Total NAP)
    for t=1:size(Trigger,2)
        [EEG_epo, epoch_start] = pop_epoch(EEG, Trigger(t),[0 300.005]); 
        
        % EOG check
        Er = EEG.data(:,floor(epoch_start):floor(epoch_start)+size(EEG_epo.data,2)-1);
        if (round(Er(1,1),2) == round(EEG_epo.data(1,1),2)) == 0
            error(msg)
        elseif (round(Er(1,end),2) == round(EEG_epo.data(1,end),2)) == 0
            error(msg)
        end
        EOG_seg = EOG(:,floor(epoch_start):floor(epoch_start)+size(EEG_epo.data,2)-1);
        
        %% 5. channel rejection
    %     EEG_epo = pop_rejchan(EEG_epo, 'elec',1:60,'threshold',5,'norm','on','measure','kurt');

        %% 6. Independent Component Analysis
        % EOG cat
        EOG_seg = cat(1, EOG_seg, mean(EOG_seg(3:4,:),1));
        EOG_seg = cat(1, EOG_seg, mean(EOG_seg(1:2,:),1)); 

        idx = [];
        iter_ica=0;
        threshold=0.7;
        EEG_epo = pop_runica(EEG_epo, 'extended',1,'interupt','off');

        ICA_Weight = EEG_epo.icaweights*EEG_epo.icasphere;
        ICA_EEG = ICA_Weight*EEG_epo.data;    

        cnt=0;
        for k=1:size(EEG_epo.data,1)
            [r,p]=corr(EOG_seg',ICA_EEG(k,:)');
            rho(:,k)=r;
            pval(:,k)=p;
            if sum(abs(r)>threshold)>=1
                cnt=cnt+1;
                idx(cnt) = k;
            end
        end  

        if isempty(idx)
            deleted_component=[];
        else
            EEG_epo = pop_subcomp( EEG_epo, idx, 0);
            deleted_component=ICA_EEG(idx,:);
        end

         %% 7. average referencing [R]
%         EEG_re=pop_reref(EEG_epo, []); %re-reference
% 
         %% 8. Laplacian filter [L]
        EEG_lap = flt_laplace('Signal',EEG_epo,'NeighbourCount',4); % laplace filter
        EEG_lap = exp_eval(EEG_lap);    

        %% 9. Epoch
        EEG_seg=eeg_regepochs(EEG_lap, 'recurrence', 30, 'limits', [0 30], 'eventtype', '30 sec'); 

        %% 10. Save
        DATA_3D = EEG_seg.data;
        DATA_2D = EEG_lap.data;
        CH = EEG_seg.chanlocs;

        save([path 'Winter_2023\Analysis\0_REST_lap\ICA_sub' num2str(n) '_ICA_' num2str(t)],'deleted_component');
        save([path 'Winter_2023\Analysis\0_REST_lap\sub' num2str(n), '_3D_' num2str(t)],'DATA_3D','NAME','CH');
        save([path 'Winter_2023\Analysis\0_REST_lap\sub' num2str(n) '_2D_' num2str(t)],'DATA_2D','NAME','CH');
        fprintf(['Sub' ,num2str(n),' Done!\n']);
    end
end