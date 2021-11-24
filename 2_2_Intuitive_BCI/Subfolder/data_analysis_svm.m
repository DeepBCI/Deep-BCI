%% Data analysis of collected EEG signal (CSP + SVM)
% The code also contains the code for CSP+RLDAshrink and CSP+RF

clear all; close all; clc; 

dd = 'datadestination';
filelist = 'filename';


%% Preprocessing cnt 
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); % Load cnt, mrk, mnt variables to Matlab

%%show eeg
%showGuiEEG(cnt,mrk)

% Parameter setting 
filtBank = [0.5 40];  % band pass filtering frequency band
ival = [-200 2000]; % time sample per epoch (considering the sampling rate)
 
% setting the subchannels for the Channel selection
  subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];

%subChannel = [57, 58, 25, 61, 62, 63, 29, 30, 31]; % visual cortex
%subChannel = [6, 44, 11, 15, 50, 16, 54, 21, 59]; % control group (randomly selected 9 channels)
%subChannel = [4, 43, 8, 13, 47, 12, 18, 52, 56]; %broca's and wernicke's area
%subChannel = [38, 5, 39, 43, 9, 10, 44, 8, 11]; % motor cortex
%subChannel = [1, 33, 34, 38, 5, 39, 35, 2, 36]; % prefrontal cortex
%subChannel = [47, 13, 48, 19, 14, 49, 15, 50, 20]; % somatosensory cortex
%subChannel = [42, 12, 17, 51, 46, 45, 16, 55, 22]; % auditory cortex

% downsampling
[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);
%[cnt, mrk] =proc_resample(cnt, 100);

% IIR Filter (Band-pass filtering)
cnt = proc_filtButter(cnt, 5, filtBank);

% Channel Selection
cnt = proc_selectChannels(cnt, subChannel);   

% MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set
mnt = mnt_adaptMontage(mnt, cnt);


%% cnt to epoch    
epo = cntToEpo(cnt, mrk, ival);

% % performing the regularization process without the baseline correction (set the parameters of the ival to -200 -> 0)
% ival22 = [0 2000];
% epo = proc_baseline(epo, ival22); 

% baseline correction from -200ms
 base = [-200 0];
 epo = proc_baseline(epo, base);
 
 ival2 = [0 2000]; 
 epo = proc_selectIval(epo,ival2);

%% Choosing the classes

% binary classification
%epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Toilet');

% whole 13 classes
%epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Clock','imagine_Hello', 'imagine_Helpme', 'imagine_Light', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Toilet', 'imagine_TV', 'imagine_Water', 'imagine_Yes', 'imagine_Rest');

% 6 classes
 epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Clock', 'imagine_Light', 'imagine_Toilet', 'imagine_TV', 'imagine_Water');
 %epo_all = proc_selectClasses(epo, 'imagine_Hello', 'imagine_Helpme', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Yes');

 % 5 classes
% epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Clock','imagine_Light','imagine_Thankyou','imagine_Toilet');


%% choosing 88 trials in each classes

% counting the numbers of each trials
count_epo=sum(epo_all.y,2);

% cutting off the trials exceeding 88 trials
y_temp = zeros(size(epo_all.y));

for ci=1:size(y_temp,1)
    cidx = find(epo_all.y(ci,:)==1);
    cidx = cidx(1:88);
    y_temp(:,cidx) = epo_all.y(:,cidx);
end

idx = find(sum(y_temp(:,:),1)==1);
y = y_temp(:,idx);
x = epo_all.x(:,:,idx);

epo_all.x = x;
epo_all.y = y;

count_epo=sum(epo_all.y,2);

%% data arrange

% arrange the y labels (changing the one hot encoding into class number labels)
y=zeros(1, 528);

for uu = 1:528
    
y(1,uu) = find(epo_all.y(:,uu)==1); % one hot encording to label

end

% separating the test and training set (10 fold cross validation)
CVO = cvpartition(y,'k',10);
%rng(1); % For reproducibility

% clear up the error rate for the evaluation
err = zeros(CVO.NumTestSets,1);

for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    
    temp_epo = epo_all;
  
    %preprocessing
    temp_epo.x = epo_all.x(:,:,trIdx);
    temp_epo.y = epo_all.y(:,trIdx);
    [fv_tr, csp_w_tr]= proc_multicsp(temp_epo,3);
    fv_tr= proc_variance(fv_tr);        
    fv_tr= proc_logarithm(fv_tr);        
    
    temp_epo.x = epo_all.x(:,:,teIdx);
    temp_epo.y = epo_all.y(:,teIdx);
    fv_te = proc_linearDerivation(temp_epo, csp_w_tr);
    fv_te = proc_variance(fv_te);
    fv_te = proc_logarithm(fv_te);
    
    %reshaping the feature vector
    train_size = sum(trIdx);
    test_size = sum(teIdx);
    
    fv_tr = reshape(fv_tr.x,[36,train_size]);% feature vector size: 36, number of trials: 476
    fv_te = reshape(fv_te.x,[36,test_size]);
    
  
%     PMd1 = fitcecoc(fv_tr',y(trIdx));
%     Md1 = PMd1.Trained{1};
    
% Performing the classification (10-fold cross validation)
% Parameters can be manipulated (see templateSVM)
    t = templateSVM('Standardize',false,'KernelFunction','RBF','KernelScale','auto');
    Md1 = fitcecoc(fv_tr',y(trIdx),'Coding','onevsall','Learners',t);
    
    ytest = predict(Md1,fv_te');
    temperr = sum((ytest'- y(teIdx))==0);
    err(i) = temperr;
    
    
end

% display the evaluation accuracy
acc = err ./ (CVO.TestSize');
a_stdev = std(acc) * 100

% display the error rate
cvErr = sum(err)/sum(CVO.TestSize);
a_ACC = cvErr*100


%% Another method (for CSP + RLDAshrink or CSP + RF)
% 
% %% Feature extraction
% 
% % basic multi-CSP
%  [csp_fv, csp_w, csp_eig] = proc_multicsp(epo_all, 3); 
% 
% %     [csp_fv, csp_w, csp_eig] = proc_csp_regularised(epo_all, 4, 1); % regulized CSP 
% %     [csp_fv, csp_w] = proc_cspscp(epo_all, 2, 1); %CSP slow cortical potential variations 
% %    [csp_fv, csp_w, csp_eig, t_filter] = proc_csssp(epo_all, 2); % Common Sparse Spectrum Spatial Pattern   
% %     [csp_fv, csp_w] = proc_cspp_auto(epo_all); %auto csp patches, only for binary-class
% 
% 
% proc= struct('memo', 'csp_w');  % number of patterns: channel number /  class number  
% proc.train= ['[fv, csp_w]= proc_multicsp(fv,3); ' ...
%             'fv= proc_variance(fv); ' ...
%             'fv= proc_logarithm(fv);'];
% proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
%             'fv= proc_variance(fv); ' ...
%             'fv= proc_logarithm(fv);'];           
% 
%         
% %% classifier: RLDAshrink / forest, 10 fold corss validation
% [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo_all, 'RLDAshrink', 'proc', proc, 'kfold', 10);
% 
% 
% 
% %% Result
% 
% % Result after cross validation = 1-error rate
% Result = 1 - C_eeg;
% Result_Std = loss_eeg_std;  
% 
% % Cross-validation result 
% Result*100
% Result_Std*100
% 
% % Confusion matrix result 
% [M, test_label] = max(out_eeg.out); % test label
% [M, true_label] = max(epo_all.y); clear M; 
% n = size(epo_all.y, 1); 
% 
% matrix_result = zeros(n, n); 
% 
% 
% for i = 1:size(test_label, 3)       
% for j = 1:length(true_label)
% matrix_result(test_label(1, j, i), true_label(j)) = matrix_result(test_label(1, j, i), true_label(j)) + 1;
% end       
% end    
% 
% 
% matrix_result = (matrix_result / sum(matrix_result(:, 1)));
% matrix_result = matrix_result * 100;
% matrix_result = matrix_result'; % true: y axis, predicted: x axis 
% 
% 
% %mmmm = round(matrix_result, 1)
% 
% %% Visualization  
% 
% %%class topographies
% % figure('Name', 'Class Topographies'); 
% % plotClassTopographies(epo_all, mnt, ival); 
% 
% %%csp patterns
% % figure('Name', 'CSP Patterns'); 
% % plotCSPatterns(csp_fv, mnt, csp_w, csp_fv.y); 
% 
% % % visualize the eeg signals
% % epo_all.x=epo_all.x(:,:,2);
% % [hp, hl]= showEEG(epo_all, ival, mrk)
% 
% % for qqq=1:64
% %     figure('Name','cnt.x(:,qqq)')
% %     plot(cnt.x(:,qqq))
% %    
% %     qqq=qqq+1
% % end
