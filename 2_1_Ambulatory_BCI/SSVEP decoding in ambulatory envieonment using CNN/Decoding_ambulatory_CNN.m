%% classifier
total_conf=zeros(3);
fileID = fopen('log_tc2.txt','w');

for sub=1:10
% data setting 
xTrain = fv_tr{sub}.x;
yTrain = categorical(fv_tr{sub}.y);
xTest = fv_te{sub}.x;
yTest = categorical(fv_te{sub}.y);

% define architecture
inputSize = [size(xTrain,1), size(xTrain,2),size(xTrain,3)];
classes = unique(yTrain);
numClasses = length(classes);

% build model
layers = build_model(inputSize, numClasses);

%% Training
options = trainingOptions('sgdm', ... %rmsprop
    'InitialLearnRate',0.01, ...
    'Verbose',false, ...
    'MaxEpochs', 500);
% 'ValidationData',{xTest,yTest}, ...
% 'Plots','training-progress', ...
net = trainNetwork(xTrain,yTrain,layers,options);

%% Test 
y_pred = classify(net,xTest);
acc(sub) = sum(double(y_pred' == yTest)/numel(yTest));
fprintf('Test accuracy s%d: %d %% \n',sub,floor(acc(sub)*100));
%% log save
t = datetime;
DateString = datestr(t);
fprintf(fileID,'%s Test accuracy s%d: %d %% \n',DateString,sub,floor(acc(sub)*100));
%%
conf{sub} = confusionmat(yTest,y_pred');
total_conf = total_conf+conf{sub};
end

fclose(fileID);

%% Functions
function layers = build_model(inputSize, outputSize)
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer([1,inputSize(2)], 32, 'padding','same') % layer3
    reluLayer
%     dropoutLayer(0.1)

    % frequency: 5.45, 8.75, 12
    convolution2dLayer([9,1], 64, 'padding','same') % layer2 [9, 1]
    reluLayer
%     dropoutLayer(0.1)

    fullyConnectedLayer(64) % layer4
    reluLayer
    dropoutLayer(0.1)
    
    fullyConnectedLayer(outputSize)
    
    softmaxLayer
    classificationLayer];
end