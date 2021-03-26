clc; clear all; close all;

folder = {'191129_lys','191129_pjy','191202_ljy','191202_ssh','191203_cjh','191206_pjh','191209_chr','191209_ykh','191210_JJH','191211_KGB','191213_MIG','191218_LCB','191219_lht','191219_yyd','191220_ydj'};
for sub=1:15
    %     sub=6;
    clearvars -except folder sub mean_R mean_L mean_ver mean_hor f_Peak
    task = 'EOG';
    cd(['D:\NEC lab\Ear-Biosignal\Data_biosignal\' folder{sub}]')
    rawdata = importdata([folder{sub} '_' task '.txt'], '\t', 2);
    
    % start tick
    fid = fopen([folder{sub} '_' task '.txt']);
    start_time = textscan(fid, '%s');
    start = cell2mat(start_time{1,1}(end-12,1));
    start = str2double(start);
    
    n = 3; Fs = 250; Fn = Fs/2;
    
    [b,a]=butter(n, [59 61]/Fn, 'stop');
    stop_data1 = filtfilt(b,a,rawdata.data);
    [b,a]=butter(n, [119 121]/Fn, 'stop');
    stop_data2 = filtfilt(b,a,stop_data1);
    [b,a]=butter(n, [0.5 124]/Fn, 'bandpass');
    bp = filtfilt(b,a,stop_data2);
    
    % cd('D:\NEC lab\Ear-Biosignal')
    % [num,txt,raw] = xlsread('EOG_information.xlsx');
    
    start2=start+250+750;
    si = 1600;
    
    for trial=1:40
        for ch=1:8
            epoch_R{ch}(:,trial) = bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),10+ch)-bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),9);
            epoch_L{ch}(:,trial) = bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),ch)-bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),19);
        end
        epoch_ver(:,trial) = bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),34);
        epoch_hor(:,trial) = bp(start2+1+1600*(trial-1):start2+si+1600*(trial-1),35);
    end
    
    task_num=[1 6 12 16 19 21 26 32 36 39 ; 3 5 9 14 17 23 25 29 34 37 ; 4 7 10 15 20 24 27 30 35 40 ; 2 8 11 13 18 22 28 31 33 38];
    
    for udrl=1:4
        for task=1:10
            for ch=1:8
                task_R{udrl,ch}(:,task) = epoch_R{ch}(:,task_num(udrl,task));
                task_L{udrl,ch}(:,task) = epoch_L{ch}(:,task_num(udrl,task));
            end
            task_ver{udrl,1}(:,task) = epoch_ver(:,task_num(udrl,task));
            task_hor{udrl,1}(:,task) = epoch_hor(:,task_num(udrl,task));
        end
    end
    
    %signal trail 평균
    for udrl=1:4
        for ch=1:8
            mean_R{udrl,ch}(:,sub) = mean(task_R{udrl,ch},2);
            mean_L{udrl,ch}(:,sub) = mean(task_L{udrl,ch},2);
        end
        mean_ver{udrl,1}(:,sub) = mean(task_ver{udrl,1},2);
        mean_hor{udrl,1}(:,sub) = mean(task_hor{udrl,1},2);
    end
    
    for udrl=1:4
        for task=1:10
            for ch=1:8
                [Mn,In] = min(task_R{udrl,ch}(1:400,task));
                [Mx,Ix] = max(task_R{udrl,ch}(1:400,task));
                value = Mx-Mn; sig=Ix-In; %양수이면 증가, 음수이면 감소
                f_Peak{sub,ch}(task,udrl) = sign(sig)*value;
                [Mn,In] = min(task_L{udrl,ch}(1:400,task));
                [Mx,Ix] = max(task_L{udrl,ch}(1:400,task));
                value = Mx-Mn; sig=Ix-In;
                f_Peak{sub,ch+8}(task,udrl) = sign(sig)*value;
            end
            [Mn,In] = min(task_ver{udrl,1}(1:400,task));
            [Mx,Ix] = max(task_ver{udrl,1}(1:400,task));
            value = Mx-Mn; sig=Ix-In;
            f_Peak{sub,17}(task,udrl) = sign(sig)*value;
            [Mn,In] = min(task_hor{udrl,1}(1:400,task));
            [Mx,Ix] = max(task_hor{udrl,1}(1:400,task));
            value = Mx-Mn; sig=Ix-In;
            f_Peak{sub,18}(task,udrl) = sign(sig)*value;                        
        end
    end
end

for sub=1:15
    for ch=1:18
        fv{sub,1}(:,ch) = [f_Peak{sub,ch}(:,1) ; f_Peak{sub,ch}(:,2)];
        fv{sub,2}(:,ch) = [f_Peak{sub,ch}(:,3) ; f_Peak{sub,ch}(:,4)];
    end
    fv{sub,1}(:,19) = [ones(10,1) ; ones(10,1)*2];
    fv{sub,2}(:,19) = [ones(10,1) ; ones(10,1)*2];
end
%%
YTrain = fv{1,1}(:,19);
YTrain2 = fv{1,2}(:,19);
addpath('D:\NEC lab\Ear-Biosignal\code\code6_filtfilt\EOG_Classify\division');
for sub=1:15
%     for ch=1:20
        for i = 1:5
%             [clasify, acc(i)] = EOG_SVM(fv{sub});
            [clasify, acc(i),YPredic(:,i)] = EOG_LDA2(fv{sub,1});
            C = confusionmat(YTrain,YPredic(:,i));
            CM(:,:,i) = C/10*100;
            
            [clasify2, acc2(i),YPredic2(:,i)] = EOG_LDA2(fv{sub,2});
            C2 = confusionmat(YTrain2,YPredic2(:,i));
            CM2(:,:,i) = C2/10*100;
        end        
        ConfusionM(:,:,sub) = mean(CM,3);
        final_accuracy(sub) = mean(acc);
        
        ConfusionM2(:,:,sub) = mean(CM2,3);
        final_accuracy2(sub) = mean(acc2);
%     end
end
CM_avr = mean(ConfusionM,3);
CM_avr2 = mean(ConfusionM2,3);
