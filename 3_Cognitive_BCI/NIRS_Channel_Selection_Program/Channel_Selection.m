clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2021\NIRS\\data\Discription');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
num_channel = 8;
    
tic
for sub = 1:15
    disp(['Subject = ' num2str(sub)])
    kfold = 10;
    load(['D:\Project\2022\NIRS\data\fv\fv_' subdir_list{sub} '.mat']);
    load('D:\Project\2022\NIRS\data\indice.mat');
    load('D:\Project\2022\NIRS\data\CH_info.mat');
    switch feattype
        case feattype == 1
            fv.x = oxy.fv.x;
            fv.y = oxy.y;
        case feattype == 2
            fv.x = deoxy.fv.x;
            fv.y = deoxy.y;
        case feattype == 3
            fv.x = hybrid.fv.x;
            fv.y = hybrid.y;
    end
            
    for k = 1:kfold
        disp(['Fold = ' num2str(k)])
        % outer
        test = (indices == k);
        train = ~test;
        fv_train.x = fv.x(:,train);
        fv_train.y = fv.y(:,train);
        fv_test.x = fv.x(:,test);
        fv_test.y = fv.y(:,test);
        
        %% 1. SFS_Op+Dist (Global ver)
        for al = 0:10
            temp_order = ch.order;
            disp(['Alpha = ' num2str(al)])
            for channel = 1:num_channel
                if channel == 1
                    for sfs = 1:length(temp_order)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{k} == Inner_k);
                            Inner_train = ~Inner_test;
                            Inner_train_x = fv_train.x(3*temp_order(sfs)-2:3*temp_order(sfs),Inner_train);
                            Inner_train_y = fv_train.y(1,Inner_train);
                            Inner_test_x = fv_train.x(3*temp_order(sfs)-2:3*temp_order(sfs),Inner_test);
                            Inner_test_y =  fv_train.y(1,Inner_test);
                            
                            Trset = train_shrinkageLDA_classifier(Inner_train_x, Inner_train_y);
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, Inner_test_x, Inner_test_y);
                            clear Inner_train_x Inner_train_y Inner_test_x Inner_test_y
                        end
                        cv_acc(sfs) = mean(c_est);
                    end
                    [~,accmax] = max(cv_acc);
                    
                    SFS(channel) = temp_order(accmax);
                    tempfeat = fv_train.x(3*temp_order(accmax)-2:3*temp_order(accmax),:);
                    temp_order(accmax) = [];
                    clear sfs c_est cv_acc accmax
                else
                    for sfs = 1:length(temp_order)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{k} == Inner_k);
                            Inner_train = ~Inner_test;
                            
                            Inner_train_x = [fv_train.x(3*temp_order(sfs)-2:3*temp_order(sfs),Inner_train); tempfeat(:,Inner_train)];
                            Inner_train_y = fv_train.y(1,Inner_train);
                            Inner_test_x  = [fv_train.x(3*temp_order(sfs)-2:3*temp_order(sfs),Inner_test); tempfeat(:,Inner_test)];
                            Inner_test_y  = fv_train.y(1,Inner_test);
                            
                            Trset = train_shrinkageLDA_classifier(Inner_train_x, Inner_train_y);
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, Inner_test_x, Inner_test_y);
                            clear Inner_train_x Inner_train_y Inner_test_x Inner_test_y
                        end
                        cv_acc(sfs) = mean(c_est);
                        temp_s = [ch.opnum(1,SFS(:)) ch.opnum(1,temp_order(sfs))];
                        temp_d = [ch.opnum(2,SFS(:)) ch.opnum(2,temp_order(sfs))];
                        optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                        
                        clear temp_s temp_d
                    end
                    clear sfs c_est
                    
                    fitness = (al/10)*(cv_acc) - (1-(al/10))*(optode/(2*channel));
                    [~,accmax] = max(fitness);
                    
                    SFS(channel) = temp_order(accmax);
                    tempfeat = [tempfeat; fv_train.x(3*temp_order(accmax)-2:3*temp_order(accmax),:)];
                    temp_order(accmax) = [];
                    clear a b optode temp_s temp_d cv_acc accmax dist
                end
            end
            Select{al+1,k} = ch.num(SFS);
            clearvars -except temp_order num_channel iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select subdir_list al temp_order
        end
    end
    save(['D:\Project\2022\NIRS_selection\Result\1101\Selected_' num2str(sub) '.mat'], 'Select', 'ch', 'indices', 'Inner_indice');
    clear ch Select indices Inner_indice
end
toc
