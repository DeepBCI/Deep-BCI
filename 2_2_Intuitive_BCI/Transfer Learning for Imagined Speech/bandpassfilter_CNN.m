
for i=1:size(eeg,3)
    e(:,:,1,i) = bandpass(eeg(:,:,i),[30 45], 1024); %gamma
end

%%
XTrain=e;
YTrain=class_word;

  %% classification
%% classication
layers = [ ...
    imageInputLayer([6 4096 1])
    convolution2dLayer([1 500],250,'Padding','same')
    convolution2dLayer([6 80],80,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.1)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

maxEpochs = 10;
miniBatchSize =32;


options = trainingOptions('adam', ...
    'InitialLearnRate',0.01001,...
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
fold=cvpartition(class_word,'kfold',kfold);
% Pre
Afold=zeros(kfold,1); confmat=0;
% Start deep learning
for i=1:kfold
  % Call index of training & testing sets
  trainIdx=fold.training(i); testIdx=fold.test(i);
  % Call training & testing features and classs
  XTrain=e(:,:,:,trainIdx); YTrain=class_word(trainIdx);
  XTest=e(:,:,:,testIdx); YTest=class_word(testIdx);
  % Convert class of both training and testing into categorial format
  YTrain=categorical(YTrain); YTest=categorical(YTest);
%   YTrain=YTrain'; YTest=YTest';
  % Training model
  net = trainNetwork(XTrain,YTrain,lgraph,options);
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
%fprintf('\n Classification Accuracy (CNN): %g %% \n ',Afold);
