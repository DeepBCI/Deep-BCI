%% Segmentation MT
clear all; close all; clc;

addpath('E:\Users\SHIN\eeglab14_1_2b\');
eeglab;

%% filename pass
% Time = {'BN', 'AN', '24H'};
Time = {'24H'};
for t = 1:size(Time,2)
    Path = ['H:\1. Journal\NeuroImage\MT_Data\' Time{t} '\'];
    DirGroup = dir(fullfile(Path,'*.vhdr'));
    FileNamesGroup = {DirGroup.name};
%% 
    for n = 1:size(FileNamesGroup,2)

    %     [EEG,c1]=pop_loadbv(Path, FileNamesGroup{n}, [], []); %EEG + EOG
        [EEG,c1]=pop_loadbv(Path, FileNamesGroup{n}, [], [1:28 33:64]); %EEG channels (1:28 33:64)   
        EEG=pop_chanedit(EEG, 'lookup','E:\\Users\\SHIN\\eeglab14_1_2b\\plugins\\dipfit2.3\\standard_BESA\\standard-10-5-cap385.elp');
        EEG=pop_resample(EEG, 250); %down-sampling
        EEG=pop_eegfiltnew(EEG, 0.5, 50); %band-pass
        EEG=pop_reref(EEG, []); %re-reference
        
%         eeglab redraw
        
        MAT.data = EEG.data;

        Tab = struct2table(EEG.event);
        Ttab = table2cell(Tab);
        MAT.MT = Ttab(:,[1 6]);

        save([['H:\1. Journal\NeuroImage\MT_Data\' Time{t} '\Data\'], ['Sub' num2str(n) '_' Time{t}]],'MAT');
    end
end
%% Procedure for preprocessing
%     Tab = struct2table(EEG.event);
%     Ttab = table2cell(Tab);
%     cnt = 0;
%     for i = 1:size(Tab,1)
%         if Ttab(i,6) == "S 22"
%             cnt = cnt+1;
%             A{cnt} = Ttab(i,1);
%         end
%             
%         if Ttab(i,6) == "S 25"
%             B = Ttab(i,1);
%         end
%     end
% %     EEG_b = pop_select(EEG,'time',[cell2mat(A{1})/1000 cell2mat(B)/1000]);   
% %     
% %     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
% %     eeglab redraw
%%
%         EEG11=pop_epoch(EEG, {'S 22'}, [0 WP_recall(i,2)]);
% 
%         EEG111=pop_select(EEG11,'trial',bb{1,n});
%         Data=EEG111.data;
%         save(['H:\수면\Nap_SM_recall\', 'Sub' num2str(n) '_BN'],'Data');
% 
%         EEG22=pop_epoch(EEG, {'S 65'}, [0.2 0.4]);
%         EEG222=pop_select(EEG22,'trial',bb{2,n});
%         Data=EEG222.data;
%         save(['H:\수면\Nap_SM_recall\', 'Sub' num2str(n) '_AN'],'Data');
%     %     save(['H:\수면\Nap_VM_recall\', 'Sub' num2str(i) '_AN'],'AN_vq1');

%% SAVE
% Data=ALLEEG(3).data;
% save(['H:\수면\Nap_SM_recall\', 'Sub7_BN_AR'],'Data');