%% 4D
eeg_4d=reshape(eeg,64, 1280, 1, 300);

%%
% XTrain=eeg_4d;
XTrain=eeg_4d;
YTrain=class';

%% classication
layers = [ ...
    imageInputLayer([64 160 1])
    convolution2dLayer([1 5],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([6 1],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 5],20,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],40,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],100,'Padding','same')
    leakyReluLayer
      convolution2dLayer([1 3],250,'Padding','same')
    leakyReluLayer
    convolution2dLayer([1 3],500,'Padding','same')
    leakyReluLayer
    dropoutLayer(0.1)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

maxEpochs = 10;
miniBatchSize =32;


options = trainingOptions('adam', ...
    'InitialLearnRate',0.001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress', 'ExecutionEnvironment','gpu' );

%%
% Start cross validation 
rng('default'); 
% Divide data into k-folds
% fold=cvpartition(label,'kfold',kfold);
kfold=5;
fold=cvpartition(class,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and classs
  XTrain=EEG_4ids(:,:,1,trainIdx); YTrain=class(trainIdx);
  XTest=EEG_4ids(:,:,1,testIdx); YTest=class(testIdx);
  % Convert class of both training and testing into categorial format
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
CNN.fold=Afold; CNN.acc=Acc; CNN.con=confmat; 
fprintf('\n Classification Accuracy (CNN): %g %% \n ',Acc);

