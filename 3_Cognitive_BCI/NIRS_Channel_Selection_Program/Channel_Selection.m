clear all
close all
clc

addpath(genpath('D:\Hyung-Tak Lee\bbci_public-master'))
addpath(genpath('D:\Hyung-Tak Lee\FDR'))
addpath(genpath('D:\Hyung-Tak Lee\RLDA'))
startup_bbci_toolbox('DataDir','D:\Hyung-Tak Lee\NIRS\data', 'TmpDir','/tmp/');
load('D:\Hyung-Tak Lee\NIRS\data\Discription.mat');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
num_channel = 15;

for sub = 1:15
    tic
    disp(['Subject = ' num2str(sub)])
    kfold = 5;
    load(['D:\Hyung-Tak Lee\NIRS\data\fv\fv_' subdir_list{sub} '.mat']);
    load('D:\Hyung-Tak Lee\NIRS\data\indice_5fold.mat');
    
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
        
        %% 1. SFS_Op+Dist (Global ver)
%         disp('SFS_Op')
        for th = 10
            for be = 10
                for al = 10
                    temp = ch.order;
%                     disp(['Alpha = ' num2str(al) ', Beta = ' num2str(be) ', Theta = ' num2str(th)])
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
                            if channel == 2
                                dist_max = max(dist);
                            end
                            fitness = (al/10)*(cv_acc) - (be/10)*(optode/(channel*2)) - (th/10)*(dist/dist_max);
                            [~,accmax] = max(fitness);
                            
                            SFS(channel) = temp(accmax);
                            tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                            temp(accmax) = [];
                            clear a b optode temp_s temp_d cv_acc accmax dist
                        end
                    end
                    Select{al+1,be+1,th+1}{k} = ch.num(SFS);
                    clearvars -except num_channel Analysis_date iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select subdir_list al be th
                end
            end
        end
    end
    toc
    save(['D:\Hyung-Tak Lee\NIRS\Result\0714\Selected_' num2str(sub) '.mat'], 'Select', 'ch', 'indices', 'Inner_indice');
    clear ch Select indices Inner_indice
end
