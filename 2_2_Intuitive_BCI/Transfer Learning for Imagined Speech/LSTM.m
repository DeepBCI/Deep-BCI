%% Reshape
eeg_2d=reshape(epo_all.x,32832, 1144);

%%
XTrain=eeg_2d;
YTrain=categorical(y);

%%
layer = lstmLayer(100,'Name','lstm1')

numFeatures = 32832;
numHiddenUnits = 20;
numClasses = 13;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 50;
miniBatchSize = 30;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);


%%
% Start cross validation 
rng('default'); 
% Divide data into k-folds
% fold=cvpartition(label,'kfold',kfold);
kfold=10;
fold=cvpartition(y,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and ys
  XTrain=eeg_2d(:,trainIdx); YTrain=y(trainIdx);
  XTest=eeg_2d(:,testIdx); YTest=y(testIdx);
  % Convert y of both training and testing into categorial format
  YTrain=categorical(YTrain); YTest=categorical(YTest);
%   YTrain=YTrain'; YTest=YTest';
  % Training model
  net = trainNetwork(XTrain,YTrain,layers,options);
  % Perform testing
  Pred=classify(net,XTest);
  % Confusion matrix
  con=confusionmat(YTest,Pred);
  % Store temporary
  confmat=confmat+con; 
  % Accuracy of each k-fold
  Afold(i,1)=100*sum(diag(con))/sum(con(:));
end
% Average accuracy over k-folds 
Acc=mean(Afold); 
% Store result
LSTM.fold=Afold; LSTM.acc=Acc; LSTM.con=confmat; 
fprintf('\n classification Accuracy (LSTM): %g %% \n ',Acc);
