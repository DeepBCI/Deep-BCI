%% EMG 분류-학습 using simple CNN 
% % 2019-07-18 by SuJin Bak
% We aim to compare classification rate between Rest state(0) and hand motion(1) using
% the EMG open data set provided by UCI.
% This is a tutorial showing the difference in 36 spectral powers.
% Reference: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures
% Accuracy: 

%clear all; clc;
% path: C:\Program Files\MATLAB\R2019a\toolbox\nnet\nndemos\nndatasets\EMG_Preprocessing
%샘플 숫자 데이터를 이미지 데이터저장소로서 불러옵니다. 
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','EMG_Preprocessing');
%imageDatastore는 폴더 이름을 기준으로 이미지에 자동으로 레이블을 지정하고,데이터를 ImageDatastore 객체로 저장합니다. 
%이미지 데이터저장소를 사용하면 메모리에 담을 수 없는 데이터를 포함하여 다량의 이미지 데이터를 저장할 수 있고,
%컨벌루션 신경망 훈련 중에 이미지 배치를 효율적으로 읽어 들일 수 있습니다.
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% 간단한 이미지 표시, 굳이 필요없음.
%figure;
%perm = randperm(36,4);
%for i = 1:4
%    subplot(2,2,i);
%    imshow(imds.Files{perm(i)});
%end


%훈련 세트의 각 범주에 5개 이미지가 포함되고 검증 세트에 각 레이블의 나머지 이미지가 포함되도록 데이터를 훈련 데이터 세트와 검증 데이터 세트로 나눕니다. 
%splitEachLabel은 데이터저장소 digitData를 2개의 새로운 데이터저장소 trainDigitData와 valDigitData로 분할합니다.
numTrainFiles = 5;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% 컨벌루션 신경망 아키텍처를 정의합니다.
%이미지 입력 계층 imageInputLayer에 이미지 크기를 지정합니다. 이 예제에서 이미지 크기는 28x28x1입니다. 각 수치는 높이, 너비, 채널 크기에 대응됩니다. 숫자 데이터는 회색조 이미지로 이루어져 있으므로 채널 크기(색 채널)는 1입니다. 컬러 이미지의 경우 채널 크기는 RGB 값에 대응하는 3입니다. trainNetwork는 기본적으로 훈련을 시작할 때 데이터를 섞기 때문에 데이터를 직접 섞지 않아도 됩니다. trainNetwork는 훈련 중에 매 Epoch가 시작할 때마다 데이터를 자동으로 섞을 수도 있습니다.

%컨벌루션 계층 컨벌루션 계층의 첫 번째 인수는 filterSize입니다. 
%이것은 이미지를 따라 스캔할 때 훈련 함수가 사용하는 필터의 높이와 너비입니다. 
%이 예제에서 3은 필터 크기가 3x3임을 나타냅니다. 필터의 높이와 너비를 다른 크기로 지정할 수 있습니다. 
%두 번째 인수는 필터의 개수 numFilters입니다. 이것은 입력의 동일한 영역에 연결되는 뉴런의 개수입니다.
%이 파라미터는 특징 맵의 개수를 결정합니다. 'Padding' 이름-값 쌍 인수를 사용하여 입력 특징 맵에 채우기를 추가합니다. 
%디폴트 스트라이드가 1인 컨벌루션 계층의 경우, 'same' 채우기를 사용하면 공간 출력 크기가 입력 크기와 같아집니다. 
%convolution2dLayer의 이름 값-쌍 인수를 사용하여 이 계층의 스트라이드와 학습률을 정의할 수도 있습니다.
%배치 정규화 계층 배치 정규화 계층은 네트워크 전체에 전파되는 활성화 값과 기울기를 정규화하여
%네트워크 훈련을 보다 쉬운 최적화 문제로 만들어 줍니다. 컨벌루션 계층과 비선형 계층(ReLU 계층 등) 사이에 배치 정규화 계층을 사용하면 네트워크 훈련 속도를 높이고 네트워크 초기화에 대한 민감도를 줄일 수 있습니다. 배치 정규화 계층은 batchNormalizationLayer를 사용하여 만듭니다.
%ReLU 계층 배치 정규화 계층 뒤에는 비선형 활성화 함수가 옵니다. 가장 자주 사용되는 활성화 함수는 ReLU(Rectified Linear Unit)입니다. ReLU 계층은 reluLayer를 사용하여 만듭니다.
%최댓값 풀링 계층 컨벌루션 계층(활성화 함수 사용)에는 특징 맵의 공간 크기를 줄여 주고 중복된 공간 정보를 제거하는 다운샘플링 연산이 뒤따르는 경우가 있습니다. 다운샘플링을 수행하면 계층당 필요한 연산량을 늘리지 않고도 보다 심층의 컨벌루션 계층에 있는 필터 개수를 늘릴 수 있습니다. 다운샘플링을 수행하는 한 가지 방법인 최댓값 풀링은 maxPooling2dLayer를 사용하여 만듭니다. 최댓값 풀링 계층은 첫 번째 인수 poolSize로 지정된 입력값이 나타내는 직사각형 영역의 최댓값을 반환합니다. 이 예제에서 직사각형 영역의 크기는 [2,2]입니다. 'Stride' 이름-값 쌍 인수는 훈련 함수가 입력값을 차례대로 스캔할 때 적용하는 스텝 크기를 지정합니다.
%완전 연결 계층 컨벌루션 계층과 다운샘플링 계층 뒤에는 하나 이상의 완전 연결 계층이 옵니다. 이름에서 알 수 있듯이 완전 연결 계층의 뉴런들은 직전 계층의 모든 뉴런에 연결됩니다. 이 계층은 이전 계층이 이미지에서 학습한 특징들을 조합하여 보다 큰 패턴을 식별합니다. 마지막 완전 연결 계층은 특징들을 조합하여 이미지를 분류합니다. 따라서 마지막 완전 연결 계층의 OutputSize 파라미터는 목표 데이터의 클래스 개수와 같습니다. 이 예제에서 출력 크기는 10개의 클래스에 대응하는 10입니다. 완전 연결 계층은 fullyConnectedLayer를 사용하여 만듭니다.
%소프트맥스 계층 소프트맥스 활성화 함수는 완전 연결 계층의 출력값을 정규화합니다. 소프트맥스 계층의 출력값은 합이 1인 양수로 구성됩니다. 이 값은 분류 계층에 의해 분류 확률로 사용될 수 있습니다. 소프트맥스 계층은 softmaxLayer 함수를 사용하여 마지막 완전 연결 계층 뒤에 만듭니다.
%분류 계층 마지막 계층은 분류 계층입니다. 이 계층은 소프트맥스 활성화 함수가 각 입력값에 대해 반환한 확률을 사용하여 상호 배타적인 클래스 중 하나에 입력값을 할당하고 손실을 계산합니다. 분류 계층을 만들려면 classificationLayer를 사용하십시오.





layers = [
    imageInputLayer([656 875 3])  
    %imageInputLayer([266 330 3])  
    %imageInputLayer([366 480 3])
    %imageInputLayer([254 355 3])
    %imageInputLayer([241 355 3])
    %imageInputLayer([416 542 3])
    %imageInputLayer([416 542 3])
    %imageInputLayer([171 267 3])
    %imageInputLayer([488 670 3])
    
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
  
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% 훈련 옵션 지정하기
%SGDM(Stochastic Gradient Descent with Momentum: 모멘텀을 사용한 확률적 경사하강법)을 사용하여 초기 학습률 0.01 네트워크를 훈련시킵니다.
%최대 Epoch 횟수를 4(original version)로 설정합니다. Epoch 1회는 전체 훈련 데이터 세트에 대한 하나의 완전한 훈련 주기를 의미. 
%검증 데이터와 검증 빈도를 지정하여 훈련 중에 네트워크 정확도를 모니터링합니다. 
%매 Epoch마다 데이터를 섞습니다. 훈련 데이터에 대해 네트워크가 훈련되고, 훈련 중에 규칙적인 간격으로 검증 데이터에 대한 정확도가 계산됩니다. 
%검증 데이터는 네트워크 가중치를 업데이트하는 데 사용되지 않습니다. 훈련 진행 상황 플롯을 켜고 명령 창 출력을 끕니다.
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


% 훈련 데이터를 사용하여 네트워크 훈련시키기
%layers에 의해 정의된 아키텍처, 훈련 데이터 및 훈련 옵션을 사용하여 네트워크를 훈련시킵니다. 
%기본적으로 trainNetwork는 GPU를 사용할 수 있으면 GPU를 사용합니다(Parallel Computing Toolbox™와 Compute Capability 3.0 이상의 CUDA® 지원 GPU 필요). 
%GPU를 사용할 수 없으면 CPU를 사용합니다. trainingOptions의 'ExecutionEnvironment' 이름-값 쌍 인수를 사용하여 실행 환경을 지정할 수도 있습니다.

%훈련 진행 상황 플롯에 미니 배치의 손실 및 정확도와 검증의 손실 및 정확도가 표시됩니다.
%훈련 진행 상황 플롯에 대한 자세한 내용은 심층 학습 훈련 진행 상황 모니터링하기 항목을 참조하십시오. 
%손실은 교차 엔트로피 손실입니다. 정확도는 네트워크가 올바르게 분류한 이미지의 비율입니다.
net = trainNetwork(imdsTrain,layers,options);


%<검증 이미지를 분류하고 정확도 계산하기>
%훈련된 네트워크를 사용하여 검증 데이터의 레이블을 예측하고 최종 검증 정확도를 계산합니다. 
%정확도는 네트워크가 올바르게 예측하는 레이블의 비율입니다. 여기서는 예측된 레이블의 99% 이상이 검증 세트의 진짜 레이블과 일치합니다.
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);


