clear all
close all
clc

%% Initial setting of the BBCI toolbox
% --- (1) specify your eeg data directory (EegMyDataDir) ------------------
addpath(genpath('D:\Project\2020\Brain_Switch\AnalysisCode\bbci_public-master_TNSRE'));

% --- (2) start the BBCI toolbox ------------------------------------------
startup_bbci_toolbox('DataDir','D:\Project\2020\Brain_Switch\Dataset\Epoch','TmpDir','/tmp/');
BTB.History = 0; % to avoid error for merging cnt

%% Preprocessing

% for feat = 1:20
for temp_subNum = 1:29
    tic
    disp(temp_subNum)
%     disp([num2str(feat) '_' num2str(temp_subNum)])
    non_csp = importdata(['D:\Project\2020\Brain_Switch\Dataset\Epoch\MA\Epo_' num2str(temp_subNum) '_non_csp.mat']);
    
    %% 10x10-fold cross-validation
    nShift = 10;
    nFold = 10;
    
    for shiftIdx = 1:nShift
        for foldIdx = 1:nFold
            load(['D:\Project\2021\DL\Brain_Switch\Data\FBCSP\10fold\CSP_' num2str(temp_subNum) '_' num2str(shiftIdx) '_' num2str(foldIdx) '.mat'])
            load(['D:\Project\2021\DL\Brain_Switch\Data\Shallow\10fold\Feature\0126\data_' num2str(temp_subNum) '_' num2str(shiftIdx) '_' num2str(foldIdx) '.mat'])
            
            % Shallow
            x_train.S.x    = double(tr_data');
            x_train.S.y    = CSP_train.y;
            
            x_test.S.x    = double(te_data');
            x_test.S.y    = CSP_test.y;
            
            y_train  = CSP_train.y;
            y_test   = CSP_test.y;
            
            % 1.CSP
            fv_train{1}.x = CSP_train.x; fv_train{1}.y = y_train; fv_train{1}.className = {'MA','Baseline'};
            fv_test{1}.x  = CSP_test.x;  fv_test{1}.y  = y_test;  fv_test{1}.className  = {'MA','Baseline'};
            
            %                 % Fisher
            %                 W = fsFisher(fv_train{1}.x', fv_train{1}.y(1,:)+1');
            %                 fv_train{1}.x = fv_train{1}.x(W.fList(1:feat),:);
            %                 fv_test{1}.x = fv_test{1}.x(W.fList(1:feat),:);
            %                 clear W
            
            % 2.Shallow
            fv_train{2}.x = x_train.S.x; fv_train{2}.y = y_train; fv_train{2}.className = {'MA','Baseline'};
            fv_test{2}.x  = x_test.S.x;  fv_test{2}.y  = y_test;  fv_test{2}.className  = {'MA','Baseline'};
            
            % 3.CSP+Shallow
            fv_train{3}.x = [fv_train{1}.x; fv_train{2}.x]; fv_train{3}.y = y_train; fv_train{3}.className = {'MA','Baseline'};
            fv_test{3}.x  = [fv_test{1}.x;   fv_test{2}.x]; fv_test{3}.y  = y_test;  fv_test{3}.className  = {'MA','Baseline'};
            
            for i = 1:3
                C{i}  = train_RLDAshrink(fv_train{i}.x, y_train);
            end
            clear i
            
            %%%%%%%%%%%%%%%%%%%%%% train meta-classifier %%%%%%%%%%%%%%%%%%
            map_train.CSP.x = apply_separatingHyperplane(C{1}, fv_train{1}.x);
            map_train.Shallow.x = apply_separatingHyperplane(C{2}, fv_train{2}.x);
            
            map_test.CSP.x = apply_separatingHyperplane(C{1}, fv_test{1}.x);
            map_test.Shallow.x = apply_separatingHyperplane(C{2}, fv_test{2}.x);
            
            % 4. meta : CSP+Shallow
            fv_train{4}.x = [map_train.CSP.x; map_train.Shallow.x];
            fv_test{4}.x  = [map_test.CSP.x;   map_test.Shallow.x];
            
            C{4}  = train_RLDAshrink(fv_train{4}.x, y_train);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            for i = 1:4
                pred = apply_separatingHyperplane(C{i}, fv_test{i}.x); % use custom function
                for nTrialN = 1:size(pred, 2)
                    if pred(nTrialN) < 0
                        pred(nTrialN) = 1;
                    elseif pred(nTrialN) > 0
                        pred(nTrialN) = 2;
                    end
                end
                acc{i}(shiftIdx,foldIdx) = mean(y_test==pred);
            end
            clear i
        end
    end
    for i = 1:4
        mean_acc{i}(temp_subNum) = mean(mean(acc{i}));
    end
    clear acc pred fv_train fv_test map_train map_test
    toc
end
%% Grand average
for i = 1:4
    GA_ACC(i, 1)  = mean(mean_acc{i});
    GA_ACC(i, 2)  = std(mean_acc{i});
end

%     c = flip(combnk(1:4,2));
%     for i = 1:size(c,1)
%         [pp(i), h(i)] = signrank(mean_acc{c(i,1)},mean_acc{c(i,2)});
%     end
%     [~,~,~,FDR] = fdr_bh(pp);
%     Feat_Acc{feat} = GA_ACC;
%     clearvars -except feat Feat_Acc
% end