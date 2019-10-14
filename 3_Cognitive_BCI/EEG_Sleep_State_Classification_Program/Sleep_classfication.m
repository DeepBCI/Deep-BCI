clc; clear;

cd('D:\NEC lab\Sleep\data\data1');
sub={'sc4002e0','sc4012e0', 'sc4102e0', 'sc4112e0', 'st7022j0', 'st7052j0', 'st7121j0', 'st7132j0'};

n=0;
for s=1:8
    tic
    n = n+1
    clearvars -except sub n s conf data hyp BTB final_accuracy_SVM2
    
    [hdr, dat] = edfread([char(sub(1,n)) '.rec']);
    [header, hypn] = edfread([char(sub(1,n)) '.hyp']);
    data{n} = dat';
    hyp{n} = hypn';
    
    for i=1:6
        st{i} = find(hyp{n}==(i-1));
        start{i} = ((st{i}-1)*3000+1);
        for trial = 1:size(start{i})
            epoch{trial,i} = data{n}(start{i}(trial,1):(start{i}(trial,1)+2999),1);
        end
    end
    if n==3
        epoch{1,5} = zeros(3000,1);
    end
    
    %% Feature
    
    Fs = 100;
    win = 100;
    overlap = 50;
    
    for i=1:6
        emptyCells{i} = cellfun(@isempty,epoch(:,i)); %# find empty cells
        conf{n,i} = size(epoch,1) - sum(emptyCells{i});
        for num = 1:conf{n,i}
            [s,f,t,p] = spectrogram(epoch{num,i}(:,1), win, overlap, [0:1:50], Fs);
            %         [s2,f2,t2,p2{num,i}] = spectrogram(epoch{num,i}(:,4), win, overlap, [0.5:1:26], Fs);
            po{i}(num,:) = mean(p,2)';
        end
    end
    
    w_Delta = mean(po{1}(:,1:5),2);
    w_Sawtooth = mean(po{1}(:,3:7),2);
    w_Theta = mean(po{1}(:,5:9),2);
    w_Alpha = mean(po{1}(:,9:13),2);
    w_Spindle = mean(po{1}(:,13:15),2);
    w_Beta = mean(po{1}(:,17:31),2);
    w_Gamma = mean(po{1}(:,32:51),2);
    
    s1_Delta = mean(po{2}(:,1:5),2);
    s1_Sawtooth = mean(po{2}(:,3:7),2);
    s1_Theta = mean(po{2}(:,5:9),2);
    s1_Alpha = mean(po{2}(:,9:13),2);
    s1_Spindle = mean(po{2}(:,13:15),2);
    s1_Beta = mean(po{2}(:,17:31),2);
    s1_Gamma = mean(po{2}(:,32:51),2);
    
    s2_Delta = mean(po{3}(:,1:5),2);
    s2_Sawtooth = mean(po{3}(:,3:7),2);
    s2_Theta = mean(po{3}(:,5:9),2);
    s2_Alpha = mean(po{3}(:,9:13),2);
    s2_Spindle = mean(po{3}(:,13:15),2);
    s2_Beta = mean(po{3}(:,17:31),2);
    s2_Gamma = mean(po{3}(:,32:51),2);
    
    s3_Delta = mean(po{4}(:,1:5),2);
    s3_Sawtooth = mean(po{4}(:,3:7),2);
    s3_Theta = mean(po{4}(:,5:9),2);
    s3_Alpha = mean(po{4}(:,9:13),2);
    s3_Spindle = mean(po{4}(:,13:15),2);
    s3_Beta = mean(po{4}(:,17:31),2);
    s3_Gamma = mean(po{4}(:,32:51),2);
    
    s4_Delta = mean(po{5}(:,1:5),2);
    s4_Sawtooth = mean(po{5}(:,3:7),2);
    s4_Theta = mean(po{5}(:,5:9),2);
    s4_Alpha = mean(po{5}(:,9:13),2);
    s4_Spindle = mean(po{5}(:,13:15),2);
    s4_Beta = mean(po{5}(:,17:31),2);
    s4_Gamma = mean(po{5}(:,32:51),2);
    
    rem_Delta = mean(po{6}(:,1:5),2);
    rem_Sawtooth = mean(po{6}(:,3:7),2);
    rem_Theta = mean(po{6}(:,5:9),2);
    rem_Alpha = mean(po{6}(:,9:13),2);
    rem_Spindle = mean(po{6}(:,13:15),2);
    rem_Beta = mean(po{6}(:,17:31),2);
    rem_Gamma = mean(po{6}(:,32:51),2);
    
    
    % mean
    for l=1:conf{n,1}
        Wake_m(l,1) = mean(epoch{l,1}(:,:));
    end
    for l=1:conf{n,2}
        S1_m(l,1) = mean(epoch{l,2}(:,:));
    end
    for l=1:conf{n,3}
        S2_m(l,1) = mean(epoch{l,3}(:,:));
    end
    for l=1:conf{n,4}
        S3_m(l,1) = mean(epoch{l,4}(:,:));
    end
    for l=1:conf{n,5}
        S4_m(l,1) = mean(epoch{l,5}(:,:));
    end
    for l=1:conf{n,6}
        Rem_m(l,1) = mean(epoch{l,6}(:,:));
    end
    
    % variance
    for l=1:conf{n,1}
        Wake_v(l,1) = var(epoch{l,1}(:,:));
    end
    for l=1:conf{n,2}
        S1_v(l,1) = var(epoch{l,2}(:,:));
    end
    for l=1:conf{n,3}
        S2_v(l,1) = var(epoch{l,3}(:,:));
    end
    for l=1:conf{n,4}
        S3_v(l,1) = var(epoch{l,4}(:,:));
    end
    for l=1:conf{n,5}
        S4_v(l,1) = var(epoch{l,5}(:,:));
    end
    for l=1:conf{n,6}
        Rem_v(l,1) = var(epoch{l,6}(:,:));
    end
    
    % skewness
    for l=1:conf{n,1}
        Wake_s(l,1) = skewness(epoch{l,1}(:,:));
    end
    for l=1:conf{n,2}
        S1_s(l,1) = skewness(epoch{l,2}(:,:));
    end
    for l=1:conf{n,3}
        S2_s(l,1) = skewness(epoch{l,3}(:,:));
    end
    for l=1:conf{n,4}
        S3_s(l,1) = skewness(epoch{l,4}(:,:));
    end
    for l=1:conf{n,5}
        S4_s(l,1) = skewness(epoch{l,5}(:,:));
    end
    for l=1:conf{n,6}
        Rem_s(l,1) = skewness(epoch{l,6}(:,:));
    end
    
    % kurtosis
    for l=1:conf{n,1}
        Wake_k(l,1) = kurtosis(epoch{l,1}(:,:));
    end
    for l=1:conf{n,2}
        S1_k(l,1) = kurtosis(epoch{l,2}(:,:));
    end
    for l=1:conf{n,3}
        S2_k(l,1) = kurtosis(epoch{l,3}(:,:));
    end
    for l=1:conf{n,4}
        S3_k(l,1) = kurtosis(epoch{l,4}(:,:));
    end
    for l=1:conf{n,5}
        S4_k(l,1) = kurtosis(epoch{l,5}(:,:));
    end
    for l=1:conf{n,6}
        Rem_k(l,1) = kurtosis(epoch{l,6}(:,:));
    end
    
    % RMS
    for l=1:conf{n,1}
        Wake_r(l,1) = rms(epoch{l,1}(:,:));
    end
    for l=1:conf{n,2}
        S1_r(l,1) = rms(epoch{l,2}(:,:));
    end
    for l=1:conf{n,3}
        S2_r(l,1) = rms(epoch{l,3}(:,:));
    end
    for l=1:conf{n,4}
        S3_r(l,1) = rms(epoch{l,4}(:,:));
    end
    for l=1:conf{n,5}
        S4_r(l,1) = rms(epoch{l,5}(:,:));
    end
    for l=1:conf{n,6}
        Rem_r(l,1) = rms(epoch{l,6}(:,:));
    end
    
    
    
    Wake_freq = [w_Delta w_Sawtooth w_Theta w_Alpha w_Spindle w_Beta];
    S1_freq = [s1_Delta s1_Sawtooth s1_Theta s1_Alpha s1_Spindle s1_Beta];
    S2_freq = [s2_Delta s2_Sawtooth s2_Theta s2_Alpha s2_Spindle s2_Beta];
    S3_freq = [s3_Delta s3_Sawtooth s3_Theta s3_Alpha s3_Spindle s3_Beta];
    S4_freq = [s4_Delta s4_Sawtooth s4_Theta s4_Alpha s4_Spindle s4_Beta];
    Rem_freq = [rem_Delta rem_Sawtooth rem_Theta rem_Alpha rem_Spindle rem_Beta];
    
    if n==3
        S4_s = 0;
        S4_k = 0;
    end
    
    Wake_time = [Wake_m Wake_v Wake_s Wake_k Wake_r];
    S1_time = [S1_m S1_v S1_s S1_k S1_r];
    S2_time = [S2_m S2_v S2_s S2_k S2_r];
    S3_time = [S3_m S3_v S3_s S3_k S3_r];
    S4_time = [S4_m S4_v S4_s S4_k S4_r];
    Rem_time = [Rem_m Rem_v Rem_s Rem_k Rem_r];
    
    fv1 = [po{1} Wake_v Wake_r ones(size(po{1},1),1); po{2} S1_v S1_r ones(size(po{2},1),1)*2; po{3} S2_v S2_r ones(size(po{3},1),1)*2; po{4} S3_v S3_r ones(size(po{4},1),1)*2;...
        po{5} S4_v S4_r ones(size(po{5},1),1)*2; po{6} Rem_v Rem_r ones(size(po{6},1),1)*2];
    fv2 = [po{1} Wake_time ones(size(po{1},1),1); po{2} S1_time ones(size(po{2},1),1)*2; po{3} S2_time ones(size(po{3},1),1)*2; po{4} S3_time ones(size(po{4},1),1)*2;...
        po{5} S4_time ones(size(po{5},1),1)*2; po{6} Rem_time ones(size(po{6},1),1)*2];
    fv3 = [w_Theta w_Alpha w_Beta Wake_v Wake_r ones(size(po{1},1),1); s1_Theta s1_Alpha s1_Beta S1_v S1_r ones(size(po{2},1),1)*2; s2_Theta s2_Alpha s2_Beta S2_v S2_r ones(size(po{3},1),1)*2;...
        s3_Theta s3_Alpha s3_Beta S3_v S3_r ones(size(po{4},1),1)*2; s4_Theta s4_Alpha s4_Beta S4_v S4_r ones(size(po{5},1),1)*2; rem_Theta rem_Alpha rem_Beta Rem_v Rem_r ones(size(po{6},1),1)*2];
    fv4 = [Wake_freq Wake_time ones(size(po{1},1),1); S1_freq S1_time ones(size(po{2},1),1)*2; S2_freq S2_time ones(size(po{3},1),1)*2; S3_freq S3_time ones(size(po{4},1),1)*2;...
        S4_freq S4_time ones(size(po{5},1),1)*2; Rem_freq Rem_time ones(size(po{6},1),1)*2];
    fv5 = [Wake_freq w_Gamma Wake_time ones(size(po{1},1),1); S1_freq s1_Gamma S1_time ones(size(po{2},1),1)*2; S2_freq s2_Gamma S2_time ones(size(po{3},1),1)*2; S3_freq s3_Gamma S3_time ones(size(po{4},1),1)*2;...
        S4_freq s4_Gamma S4_time ones(size(po{5},1),1)*2; Rem_freq rem_Gamma Rem_time ones(size(po{6},1),1)*2];
    fv6 = [po{1} ones(size(po{1},1),1); po{2} ones(size(po{2},1),1)*2; po{3} ones(size(po{3},1),1)*2; po{4} ones(size(po{4},1),1)*2;...
        po{5} ones(size(po{5},1),1)*2; po{6} ones(size(po{6},1),1)*2];
    fv7 = [w_Theta w_Alpha w_Beta ones(size(po{1},1),1); s1_Theta s1_Alpha s1_Beta ones(size(po{2},1),1)*2; s2_Theta s2_Alpha s2_Beta ones(size(po{3},1),1)*2;...
        s3_Theta s3_Alpha s3_Beta ones(size(po{4},1),1)*2; s4_Theta s4_Alpha s4_Beta ones(size(po{5},1),1)*2; rem_Theta rem_Alpha rem_Beta ones(size(po{6},1),1)*2];
    fv8 = [w_Theta w_Alpha w_Spindle w_Beta ones(size(po{1},1),1); s1_Theta s1_Alpha s1_Spindle s1_Beta ones(size(po{2},1),1)*2; s2_Theta s2_Alpha s2_Spindle s2_Beta ones(size(po{3},1),1)*2;...
        s3_Theta s3_Alpha s3_Spindle s3_Beta ones(size(po{4},1),1)*2; s4_Theta s4_Alpha s4_Spindle s4_Beta ones(size(po{5},1),1)*2; rem_Theta rem_Alpha rem_Spindle rem_Beta ones(size(po{6},1),1)*2];
    fv9 = [w_Delta w_Theta w_Alpha w_Spindle w_Beta ones(size(po{1},1),1); s1_Delta s1_Theta s1_Alpha s1_Spindle s1_Beta ones(size(po{2},1),1)*2; s2_Delta s2_Theta s2_Alpha s2_Spindle s2_Beta ones(size(po{3},1),1)*2;...
        s3_Delta s3_Theta s3_Alpha s3_Spindle s3_Beta ones(size(po{4},1),1)*2; s4_Delta s4_Theta s4_Alpha s4_Spindle s4_Beta ones(size(po{5},1),1)*2; rem_Delta rem_Theta rem_Alpha rem_Spindle rem_Beta ones(size(po{6},1),1)*2];
    fv10 = [Wake_freq ones(size(po{1},1),1); S1_freq ones(size(po{2},1),1)*2; S2_freq ones(size(po{3},1),1)*2; S3_freq ones(size(po{4},1),1)*2;...
        S4_freq ones(size(po{5},1),1)*2; Rem_freq ones(size(po{6},1),1)*2];
    fv11 = [Wake_freq w_Gamma ones(size(po{1},1),1); S1_freq s1_Gamma ones(size(po{2},1),1)*2; S2_freq s2_Gamma ones(size(po{3},1),1)*2; S3_freq s3_Gamma ones(size(po{4},1),1)*2;...
        S4_freq s4_Gamma ones(size(po{5},1),1)*2; Rem_freq rem_Gamma ones(size(po{6},1),1)*2];
    
    
    %% classification
    
    
    iterNum = 10;
    
    for type = 1:11
        clear predictorNames
        fv = eval(['fv' num2str(type)]);
        for i = 1:iterNum
            
            trainingData = fv(:,:);
            
            Names = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12',...
                'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25',...
                'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38',...
                'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51',...
                'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57'};
            
            for ii=1:size(trainingData,2)
                predictorNames(ii) = Names(ii);
            end
            
            inputTable = array2table(trainingData, 'VariableNames', predictorNames);
            
            predictors = inputTable(:, predictorNames(:,1:end-1));
            response = trainingData(:,end);
            isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
            
            classificationSVM = fitcsvm(...
                predictors, ...
                response, ...
                'KernelFunction', 'polynomial', ...
                'PolynomialOrder', 2, ...
                'KernelScale', 'auto', ...
                'BoxConstraint', 1, ...
                'Standardize', true, ...
                'ClassNames', [1; 2]);
            
            predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
            svmPredictFcn = @(x) predict(classificationSVM, x);
            trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));
            trainedClassifier.ClassificationSVM = classificationSVM;
                                   
            partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);
            
            [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
            
            svm2_acc(i) = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
        end
        
        final_accuracy_SVM2(type,n) = mean(svm2_acc);
        
    end
    toc
end

