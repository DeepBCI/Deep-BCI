% SVM_Clasification.m
% 
% Subject dependent classification using SVM
% 
% author: Young-Seok, Kweon
% created: 2019.07.15

%% initialize

clear;close all; 
startup_bbci_toolbox
clc;

%% set value for data load

basic_dir='E:\data_result\';
Type_filename={'High_P_','Medium_P_','Low_P_';... % there were three groups according to dosage
    'High_M_','Medium_M_','Low_M_'}; % there were two groups according to the kind of agnets (Midazolam and Propofol)
Type={'P_Result_','M_Result_'};
num_agents=size(Type_filename,1); % number of agents
num_states=size(Type_filename,2); % number of dosage groups

name_part='DataSet_All';
frequency={'delta','theta','alpha','beta','gamma','raw'};
num_frequency=size(frequency,2);

%% data load

for i=1:num_agents
    for j=1:num_states
        fprintf('******************%s******************\n',Type_filename{i,j});
        tic;
        name=strcat(Type_filename{i,j},name_part);
        data=load(strcat(basic_dir,name));
        
        data=data.data; % get eeg signals 
        y=logical(data.y); % get lable of subjects
        time=toc;
        calculate_time(time);
        fprintf('[Data Load Done]\n');
        for k=1:num_frequency
            [X,Y]=preprocessing(data.x,frequency{k},y);
            for fold=1:10
                [XTrain, YTrain, XTest,YTest]=segmentation(X,Y,fold);
                
                tic;
               
                svmmodel=fitcsvm(XTrain,YTrain','KernelFunction','RBF','KernelScale','auto'); % change the 'RBF' to 'linear' if you want to run linear SVM
                time=toc;                
                calculate_time(time);
                
                [YPredicted,score] = predict(svmmodel,XTest);
                % To know the true positive, true negative, false positive, and false negative
                pp(k,fold)=0;pn(k,fold)=0;nn(k,fold)=0;np(k,fold)=0;
                 for num=1:30
                     if YTest(num)==categorical({'1'})
                         pp(k,fold)=pp(k,fold)+(YPredicted(num)==YTest(num));
                         pn(k,fold)=pn(k,fold)+~(YPredicted(num)==YTest(num));
                     else
                         nn(k,fold)=nn(k,fold)+(YPredicted(num)==YTest(num));
                         np(k,fold)=np(k,fold)+~(YPredicted(num)==YTest(num));
                     end
                 end
%                  fprintf('P/A  Pos Neg\n');
%                  fprintf('Pos   %d  %d\n',pp(k,fold),np(k,fold));
%                  fprintf('Neg   %d  %d\n',pn(k,fold),nn(k,fold));
                accuracy(k,fold)=sum(trace(YTest==YPredicted))/30;
                fprintf('(%d) accuracy: %f\n',fold,accuracy(fold));
            end
            temp=mean(accuracy,2);
            fprintf('%s: average accuracy: %f\n',frequency{k},temp(k));
%             all(j,k)=mean(accuracy);
        end
        save(strcat(strcat(basic_dir,Type_filename{i,j}),...
            'SVM_rbf'),'accuracy','pp','np','nn','pn');
    end
    
end
%% segmentation data
function [XTrain, YTrain, XTest, YTest]=segmentation(X,Y,k)
trials=30;
idx=[];
for i=1:10
    idx=[idx (trials*(i-1)+3*(k-1)+1):(trials*(i-1)+3*k)];
end
% LOO
% idx=(trials*(k-1)+1):(trials*k);

XTest=X(idx,:);
YTest=Y(idx);
% for i=1:size(YTest,2)
%     temp(i,YTest(i)+1)=1;
% end
YTest=categorical(YTest);

temp=X;
temp(idx,:)=[];
XTrain=temp;

temp=Y;
temp(idx)=[];
% for i=1:size(temp,2)
%     temp_1(i,temp(i)+1)=1;
% end
YTrain=categorical(temp);

end
%% preprocessing data for SVM  
function [signal,label]=preprocessing(x,name,label)

% make 4000xnum_ch flatten (time x channel)
eval(['signal=x.' name ';']); 
channel=1:62;
num_ch=size(channel,2);
temp=[];
for i=channel
    temp=cat(1,temp,signal(:,i,:,:));
end
signal=reshape(temp,[4000*num_ch 30 10]);
% make 30x10 flatten (trials x subjects)
temp=[];
num_sub=size(signal,3);
for i=1:num_sub
    temp=cat(2,temp,signal(:,:,i));
end
signal=temp';
% make label
temp=[];
for i=1:num_sub
    extend=ones([1 30])*label(i);
    temp=[temp extend];
end
label=temp;
end
%% calculate time
function calculate_time(time)
    hour=floor(time/3600);
    time=time-hour*3600;
    min=floor(time/60);
    time=time-min*60;
    sec=floor(time);
    fprintf('Training Time: %d ½Ã°£ %d ºÐ %d ÃÊ \n',hour,min,sec);
end
