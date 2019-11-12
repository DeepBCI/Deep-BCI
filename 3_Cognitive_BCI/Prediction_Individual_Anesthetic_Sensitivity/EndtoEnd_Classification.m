% EndtoEnd_Classification.m:
% 
% EndtoEnd CNN으로 High sensitivity group인지 Low sensitivity group인지
% classification 한다.
% 
% High_P_CNN_Result.m으로 저장
% High_M_CNN_Result.m으로 저장
% 
% author: Young-Seok Kweon
% created: 2019.06.18
%% main
clc;clear; close all;
frequency={'delta','theta','alpha','beta','gamma'};
for i=1:1
	CNN(frequency{i});
end

%% function 
function CNN(frequency)
%% value setting for data setting

basic_dir='E:\data_result\';
Type_filename={'High_P_','Medium_P_','Low_P_'};
% Type_filename={'High_M_','Medium_M_','Low_M_'};
name_s={'DataSet_All'};
n=10;

channel=1:62;
num_ch=size(channel,2);

%% CNN layer construction
        
layers = [imageInputLayer([4000 62 1])

convolution2dLayer([1*10 62],62) % conv1 layer to integrate the information of channel
batchNormalizationLayer % To improve performance of layer, use batch normalization
reluLayer

convolution2dLayer([1*10 1],124) % conv2 layer to integrate the information of time
batchNormalizationLayer % To improve performance of layer, use batch normalization
reluLayer

% averagePooling2dLayer([2 2],'Stride',2)
maxPooling2dLayer([2 1],'Stride',2) 

fullyConnectedLayer(2) % 2 unit for one which is high sensitivity and the other which is low sensitivity
softmaxLayer
classificationLayer];

index=1;
%% data setting and load


for state=1:size(Type_filename,2)
    
    fprintf('**************%s****************',Type_filename{state});
    name=strcat(Type_filename{state},name_s{1});
    data=load([basic_dir name]);
    data=data.data;
    
    % data reshape
    X=[];Y=[];
    n=size(data.y,1);
    trials=size(data.x.raw,3);
    for i=1:n
        eval(['temp=data.x.' frequency '(:,:,:,i);']);
        temp=temp(:,channel,:,:);
        X=cat(3,X,temp);
        Y=cat(2,Y,ones([1,trials])*data.y(i));
    end
    X=reshape(X,[4000 num_ch 1 300]);
%% Cross validation

    fold=10;temp=[];
    for i=1:fold
        % split into training and testing
        idx=zeros([1,trials*n]);
        for j=1:n
            temp=trials*(j-1)+trials/fold*(i-1);
            idx(temp+1:temp+trials/fold)=1;
        end
         % Leave Out One
%         idx=zeros([1,trials*n]);
%         idx(1+(i-1)*trials:i*trials)=1;
%         
        XVal=X(:,:,:,idx==1);
        XTrain=X(:,:,:,idx==0);
        
        YVal=Y(idx==1);
        YVal=categorical(cellstr(num2str(YVal'))); % trainNetwork 함수에 y는 categorical로 들어가야함
        YTrain=Y(idx==0);
        YTrain=categorical(cellstr(num2str(YTrain'))); % trainNetwork 함수에 y는 categorical로 들어가야함
        
        options = trainingOptions('sgdm', ... % training method is stochastic gradient descent method
            'MaxEpochs',30, ... % maximum numbers of epochs
            'InitialLearnRate',1e-3, ...
            'MiniBatchSize',30,...
            'Verbose',false); % whether show the progress of training

        net = trainNetwork(XTrain,YTrain,layers,options); % training the network
        
        fprintf('\nFold %d\n',i);
        predicted_train = classify(net,XTrain); % classify training data using trained net
        accuracy_train(i) = sum(predicted_train == YTrain)/numel(YTrain)*100; 
        fprintf('Training accuracy: %f\n',accuracy_train(i));

        predicted_test = classify(net,XVal); % classify validation data using trained net
        accuracy_test(i) = sum(predicted_test == YVal)/numel(YVal)*100;
        fprintf('Testing accuracy: %f\n',accuracy_test(i));
        
        w = warning('query','last');
        id = w.identifier;
        warning('off',id)
        rmpath('folderthatisnotonpath')
    end
    
    fprintf('********************************\nAverage Training Accuracy: %f\n:'...
        ,sum(accuracy_train)./fold);
    fprintf('********************************\nAverage Testing Accuracy: %f\n:'...
        ,sum(accuracy_test)./fold);
	
     title=strcat(...
         strcat('E:\data_result\',Type_filename{state}),...
         'CNN_Result_2Layer_',frequency,'_LOO_',num2str(index)); 
     
     save(title,'accuracy_train','accuracy_test'); % save the training and testing accuracy 
    
end

clc;clear; close all;
end
