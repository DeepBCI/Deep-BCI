clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2021\NIRS\20210203\data\Discription.mat');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
num_channel = 15;

for sub = 1:15
    tic
    disp(['Subject = ' num2str(sub)])
    kfold = 10;
    load(['D:\Project\2021\NIRS\20210203\data\fv\fv_' subdir_list{sub} '.mat']);
    load('D:\Project\2021\NIRS\20210203\data\indice.mat');
    
    for search = 1:size(fv.clab,2)
        ch.num(1,search) = sscanf(cell2mat(fv.clab(search)),strcat("CH","%d","oxy"));
        ch.opnum(:,search) = op_info(ch.num(1,search),2:3)';
        [ch.cyloc(1,search), ch.cxloc(1,search)] = find(op_loc == ch.num(1,search));
        [ch.syloc(1,search), ch.sxloc(1,search)] = find(op_s == ch.opnum(1,search));
        [ch.dyloc(1,search), ch.dxloc(1,search)] = find(op_d == ch.opnum(2,search));
    end
    ch.order = 1:size(fv.clab,2);
    
    for k = 1:kfold
        disp(['Fold = ' num2str(k)])
        % outer
        test = (indices == k);
        train = ~test;
        fv_train.x = fv.x(:,train);
        fv_train.y = fv.y(:,train);
        fv_test.x = fv.x(:,test);
        fv_test.y = fv.y(:,test);
        
        %% SFS_Alpha_Beta_Gamma
        disp('SFS_Alpha_Beta_Gamma')
        for gamma = 0:10
            for beta = 0:10
                for alpha = 0:10
                    disp(['Alpha = ' num2str(alpha/10) ', Beta = ' num2str(beta/10) ', Gamma = ' num2str(beta/10)]);
                    temp = ch.order;
                    for channel = 1:num_channel
                        if channel == 1
                            for sfs = 1:length(temp)
                                for Inner_k = 1:kfold
                                    Inner_test = (Inner_indice{k} == Inner_k);
                                    Inner_train = ~Inner_test;
                                    Trset = train_shrinkageLDA_classifier(fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train), fv_train.y(1,Inner_train));
                                    c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test), fv_train.y(1,Inner_test));
                                end
                                cv_acc(sfs) = mean(c_est);
                            end
                            [~,accmax] = max(cv_acc);
                            
                            SFS(channel) = temp(accmax);
                            tempfeat = fv_train.x(3*temp(accmax)-2:3*temp(accmax),:);
                            temp(accmax) = [];
                            clear sfs c_est cv_acc accmax
                        else
                            for sfs = 1:length(temp)
                                for Inner_k = 1:kfold
                                    Inner_test = (Inner_indice{k} == Inner_k);
                                    Inner_train = ~Inner_test;
                                    Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                                    c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                                end
                                cv_acc(sfs) = mean(c_est);
                                temp_s = [ch.opnum(1,SFS(:)) ch.opnum(1,temp(sfs))];
                                temp_d = [ch.opnum(2,SFS(:)) ch.opnum(2,temp(sfs))];
                                optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                                
                                loc1 = mean([ch.cxloc(SFS(:)); ch.cyloc(SFS(:))]',1);
                                loc2 = [ch.cxloc(temp(sfs)) ch.cyloc(temp(sfs))];
                                dist(sfs) = pdist([loc1; loc2]);
                                clear loc1 loc2 dist_temp temp_s temp_d
                            end
                            clear sfs c_est
                            
                            fitness = (alpha/10) * cv_acc - (beta/10)*(optode/52) - (gamma/10)*(dist/24.08);
                            [~,accmax] = max(fitness);
                            dist_indi(channel) = dist(accmax)*(channel-1);
                            
                            SFS(channel) = temp(accmax);
                            tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                            temp(accmax) = [];
                            clear a b optode temp_s temp_d cv_acc accmax dist
                        end
                    end
                    Select{2}{k}{alpha+1,beta+1,gamma+1} = ch.num(SFS);
                    clearvars -except num_channel Analysis_date iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select subdir_list alpha beta gamma
                end
            end
        end
        clearvars -except num_channel Analysis_date iter fv_train Inner_indice sub ch fv indices Inner_indice kfold k op_d op_s op_info op_loc opx opy Select sub tempfeat subdir_list fv_train
    end
    toc
    save(['D:\Project\2021\NIRS\20210203\Result\Selected_' num2str(sub) '.mat'], 'Select', 'ch', 'indices', 'Inner_indice');
    clear ch Select indices Inner_indice
end
