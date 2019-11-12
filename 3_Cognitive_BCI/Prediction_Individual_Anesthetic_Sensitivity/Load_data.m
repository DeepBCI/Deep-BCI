% Load_data.m: Load excel and get subject information. This script
% separates subjects into 6 groups. First criteria is a kind of
% anesthetics,and there are 2 anesthetics.
% 
% Propofol/Midazolam
% 
% Second criteria is a dose of anesthetics, and there are 3 level of dose.
% 
% High/Medium/Low
% 
% Therefore there are 6 groups. 
% 
% 
% created: 2019.05.07
% author: Young-Seok, Kweon

%% initialize

clc;clear;close all;
%% load excel 
tic;
n=60; % number of subjects
dir='E:\Sedation raw data (n=60)\';
filename='Subject_info';
CeData='CeData\';
agents={'PPF\','MDZ\'};
[num,txt,raw]=xlsread(strcat(dir,filename));
fprintf('Excel Load: ');
toc

%% excel data processing

tic;
type=[2,7,8]; % excel 파일에서 가져올 column 지정
%     2: identity code
%     7: kind of anesthetic
%     8: dose of anesthetic

anesthetic=txt(:,type); % select what we want
anesthetic(1,:)=[];
identity{n}=0;

basic_name='anesthesia_';

for i=1:n % make filename to fit the eeg file
    temp=strsplit(string(anesthetic(i,1)),'_');
    temp=temp([1 3]);
    temp=strcat(strcat(temp(2),'_'),temp(1));
    anesthetic(i,1)=cellstr(strcat(basic_name,temp));
end

% separate subjects (propofol/midazolam)
index_P=find(char(anesthetic(:,2))=='P');
index_M=find(char(anesthetic(:,2))=='M');

group{1}=anesthetic(index_P,:);
group{2}=anesthetic(index_M,:);
group_final{2,3}="abcd";
% separate subjects (High/Medium/Low)
for i=1:2 
    index_H=find(char(group{i}(:,3))=='H');
    index_M=find(char(group{i}(:,3))=='M');
    index_L=find(char(group{i}(:,3))=='L');
   
    group_final{i,1}=group{i}(index_H,:);
    group_final{i,2}=group{i}(index_M,:);
    group_final{i,3}=group{i}(index_L,:);
end
fprintf('Info processing: ');
toc
%% file name check
kind={'P_','M_'};
dose={'High_','Medium_','Low_'}; % set the name for specific dose
Ce_name{size(kind,2),size(dose,2)}(10)="abcd"; % 사전할당

for i_kind=1:size(kind,2)
    for i_dose=1:size(dose,2)
        for i=1:10
            % EEG 확인
            a=strcat(strcat(dir,'EEGData\'),group_final{i_kind,i_dose}(i));
            a=string(a);
            b=group_final{i_kind,i_dose}(i);
            b=strcat(string(b),'.eeg');
            if isfile(strcat(strcat(a,'\'),b))
            else
                fprintf('%s is wrong\n',b);
            end
            
            % xls 확인
         
            if i_kind==1
                %PPF
                final_dir=strcat(strcat(dir,CeData),agents{i_kind});

                sub_name=strcat(group_final{i_kind,i_dose}(i,2),... % P/M
                    group_final{i_kind,i_dose}(i,3)); % H/M/L

                if ~(i==10)
                    sub_final_name=strcat(...
                        strcat(sub_name,strcat(num2str(0),num2str(i))),...
                        '_');
                else
                    sub_final_name=strcat(...
                        strcat(sub_name,num2str(i)),...
                        '_');
                end

                final_name=strcat(sub_final_name,group_final{i_kind,i_dose}(i));
                a=final_dir;
                a=string(a);
                b=strcat(string(final_name),'.xlsx');
                Ce_name{i_kind,i_dose}(i)=strcat(strcat(a,'\'),b);
                if isfile(Ce_name{i_kind,i_dose}(i))
                else
                    fprintf('%s is wrong\n',b);
                end
            else
                %MDZ
                final_dir=strcat(strcat(dir,CeData),agents{i_kind});
                
                temp=strsplit(string(group_final{i_kind,i_dose}(i)),'_');
                temp_char=char(temp(3));
                
                final_name=strcat(...
                    strcat(temp_char(1),temp_char(6:end)),...
                    '통합결과2');
                
                a=final_dir;
                a=string(a);
                b=strcat(string(final_name),'.xlsx');
                Ce_name{i_kind,i_dose}(i)=strcat(a,b);
                
                if isfile(Ce_name{i_kind,i_dose}(i))
                else
                    fprintf('%s is wrong\n',b);
                end
            end
        end
    end
end
fprintf('\n\n');
%% EEG data load and convert

startup_bbci_toolbox
sampling_frequency=1000;
time=4;
dir_save='E:\data_convert\';

for i_kind=1:size(kind,2)
    for i_dose=1:size(dose,2)
        for i=1:10
            fprintf('%s',string(dose(i_dose)));
            fprintf('%s',string(kind(i_kind)));
            fprintf("%d 번째 ",i);
            tic;
            % File 경로설정
            BTB.DataDir=strcat(strcat(dir,'EEGData\'),group_final{i_kind,i_dose}(i));
            BTB.filename= group_final{i_kind,i_dose}(i);
            BTB.RawDir=fullfile(BTB.DataDir, BTB.filename);
            % eeg 읽기
            [cnt, mrk_orig, hdr] =file_readBV(BTB.RawDir,'Fs',sampling_frequency);
            % excel 읽기
            [num_ce,txt_ce,raw_ce]=xlsread(Ce_name{i_kind,i_dose}(i),'CpCe');
            [num_bis,txt_bis,raw_bis]=xlsread(Ce_name{i_kind,i_dose}(i),'BIS');
            
            % Butterworth filter
            band= [2 250];

            [numerical,decimeter]= butter(5, band/cnt.fs*2);
            cnt= proc_filtfilt(cnt, numerical, decimeter);

            % Segmentation
            disp_ival= [0 sampling_frequency*time]; % 0 s ~ time s
            
            trig= {1,2, 3, 4,8; ...
                'Beginning Point','Baseline Stimuls','Sedation Stimuls','End Point','Button Press'};
            
            trig_hj={11,1, 4,8; ...
                'Beginning Point','Stimuls','End Point','Button Press'};
            
            if i_kind==1 && i_dose==2 && i==1
                mrk= mrk_defineClasses(mrk_orig, trig_hj);
            else
                mrk= mrk_defineClasses(mrk_orig, trig);
            end
            
            cnt = proc_segmentation(cnt, mrk, disp_ival);
            
            cnt.cpce = num_ce;
            cnt.bis = raw_bis;
            
            % Set mnt
            mnt.x=num_ce;
            mnt.y=cnt.y;
            mnt.clab=cnt.clab;

            % Convert the .eeg raw data file to .mat file

            cnt.title=char(strcat(...
                strcat(dose(i_dose),kind(i_kind))...
                ,num2str(i)));
           
            file_saveMatlab(strcat(dir_save,cnt.title), cnt, mrk, mnt, ...
                'channelwise',1, ...
                'format','int16', ...
                'resolution', NaN);
            
            w=warning('query','last'); % 경고 무시
            warning('off',w.identifier)

            toc;
        end
    end
end
