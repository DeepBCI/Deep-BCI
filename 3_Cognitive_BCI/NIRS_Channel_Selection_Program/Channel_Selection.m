clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2020\NIRS\Channel_selection\20201019\data\op_info');
load('D:\Project\2020\NIRS\Channel_selection\20201019\data\op_loc');
load('D:\Project\2020\NIRS\Channel_selection\20201019\data\sd_loc');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
num_channel = 20;


for sub = 1:15
    disp(['Subject = ' num2str(sub)])
    iter = 10;
    kfold = 5;
    load(['D:\Project\2020\NIRS\Channel_selection\20201019\data\fv\fv_' subdir_list{sub} '.mat']);
    
    for search = 1:size(fv.clab,2)
        ch{sub}.num(1,search) = sscanf(cell2mat(fv.clab(search)),strcat("CH","%d","oxy"));
        ch{sub}.opnum(:,search) = op_info(ch{sub}.num(1,search),2:3)';
        [ch{sub}.cyloc(1,search), ch{sub}.cxloc(1,search)] = find(op_loc == ch{sub}.num(1,search));
        [ch{sub}.syloc(1,search), ch{sub}.sxloc(1,search)] = find(op_s == ch{sub}.opnum(1,search));
        [ch{sub}.dyloc(1,search), ch{sub}.dxloc(1,search)] = find(op_d == ch{sub}.opnum(2,search));
    end
    ch{sub}.order = 1:size(fv.clab,2);
    
    for m = 1:iter
        tic
        indices{sub}{m} = crossvalind('Kfold', fv.y(1, :)', kfold);
        for k = 1:kfold
            disp(['Iter = ' num2str(m) ', Fold = ' num2str(k)])
            %% outer
            test = (indices{sub}{m} == k);
            train = ~test;
            fv_train.x = fv.x(:,train);
            fv_train.y = fv.y(:,train);
            fv_test.x = fv.x(:,test);
            fv_test.y = fv.y(:,test);
            Inner_indice{sub}{m} = crossvalind('Kfold', fv_train.y(1, :)', kfold);
            
            %% 1. fisher
            
            disp('Fisher')
            W = fsFisher(fv.x(:,train)', fv.y(1,train)'); %feature º±≈√
            
            for i = 1:size(fv.clab,2)
                fisher_weight(i) = sum(W.W((3*i-2):3*i));
            end
            [~,fList] = sort(fisher_weight, 'descend');
            Select{1}{sub}{m}{k} = ch{sub}.num(fList(1:20));
            
            %% 2. SFS
            
            disp('SFS')
            temp = ch{sub}.order;
            for channel = 1:num_channel
                if channel == 1
                    for sfs = 1:length(temp)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
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
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
                            Inner_train = ~Inner_test;
                            Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                        end
                        cv_acc(sfs) = mean(c_est);
                    end
                    
                    [~,accmax] = max(cv_acc);
                    
                    SFS(channel) = temp(accmax);
                    tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                    temp(accmax) = [];
                    clear sfs c_est cv_acc accmax
                end
            end
            Select{2}{sub}{m}{k} = ch{sub}.num(SFS);
            clearvars -except num_channel Analysis_date m iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select sub tempfeat subdir_list
            
            %% 3. SFS_Op (Flexible)
            
            disp('SFS_Op (Individual)')
            temp = ch{sub}.order;
            for channel = 1:num_channel
                if channel == 1
                    for sfs = 1:length(temp)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
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
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
                            Inner_train = ~Inner_test;
                            Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                        end
                        cv_acc(sfs) = mean(c_est);
                        temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                        temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                        optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                        
                        clear loc1 loc2 dist_temp
                    end
                    clear sfs c_est
                    
                    fitness = cv_acc - (optode/(2*channel));
                    [~,accmax] = max(fitness);
                    
                    SFS(channel) = temp(accmax);
                    tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                    temp(accmax) = [];
                    clear a b optode cv_acc accmax dist
                end
            end
            Select{3}{sub}{m}{k} = ch{sub}.num(SFS);
            clearvars -except num_channel Analysis_date m iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select sub tempfeat subdir_list
            
            %% 4. SFS_Op (Max)
            
            disp('SFS_Op (Global)')
            temp = ch{sub}.order;
            for channel = 1:num_channel
                if channel == 1
                    for sfs = 1:length(temp)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
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
                            Inner_test = (Inner_indice{sub}{m} == Inner_k);
                            Inner_train = ~Inner_test;
                            Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                        end
                        cv_acc(sfs) = mean(c_est);
                        temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                        temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                        optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                        
                        clear temp_s temp_d
                    end
                    clear sfs c_est
                    
                    fitness = cv_acc - (optode/52);
                    [~,accmax] = max(fitness);
                    
                    SFS(channel) = temp(accmax);
                    tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                    temp(accmax) = [];
                    clear a b optode temp_s temp_d cv_acc accmax dist
                end
            end
            Select{4}{sub}{m}{k} = ch{sub}.num(SFS);
            clearvars -except num_channel Analysis_date m iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select sub tempfeat subdir_list
            
            %% 5~8. SFS_Op (Flexible, alpha)
            
            for alpha = 6:9
                disp(['SFS_Op (Individual) _ ' num2str(alpha/10)])
                temp = ch{sub}.order;
                for channel = 1:num_channel
                    if channel == 1
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub}{m} == Inner_k);
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
                                Inner_test = (Inner_indice{sub}{m} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                            temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                            temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                            optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                            
                            clear loc1 loc2 dist_temp
                        end
                        clear sfs c_est
                        
                        fitness = ((1 - (alpha/10)) * cv_acc) - (alpha/10 * (optode/(2*channel)));
                        [~,accmax] = max(fitness);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                        temp(accmax) = [];
                        clear a b optode cv_acc accmax dist
                    end
                end
                Select{alpha}{sub}{m}{k} = ch{sub}.num(SFS);
                clearvars -except num_channel Analysis_date m iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select sub tempfeat subdir_list
            end
            
            %% 9~12. SFS_Op (Max, beta)
            
            for beta = 6:9
                disp(['SFS_Op (Global) _ ' num2str(beta/10)])
                temp = ch{sub}.order;
                for channel = 1:num_channel
                    if channel == 1
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub}{m} == Inner_k);
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
                                Inner_test = (Inner_indice{sub}{m} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier([fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], fv_train.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [fv_train.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], fv_train.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                            temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                            temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                            optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                            
                            clear temp_s temp_d
                        end
                        clear sfs c_est
                        
                        fitness = ((1 - (beta/10)) * cv_acc) - (beta/10 * (optode/52));
                        [~,accmax] = max(fitness);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = [tempfeat; fv_train.x(3*temp(accmax)-2:3*temp(accmax),:)];
                        temp(accmax) = [];
                        clear a b optode temp_s temp_d cv_acc accmax dist
                    end
                end
                Select{beta+5}{sub}{m}{k} = ch{sub}.num(SFS);
                clearvars -except num_channel Analysis_date m beta iter fv_train Inner_indice sub ch fv indices kfold k op_d op_s op_info op_loc opx opy Select sub subdir_list
            end
        end
        toc
    end
end

save('D:\Project\2020\NIRS\Channel_selection\20201019\Result\Select.mat', 'Select');
save('D:\Project\2020\NIRS\Channel_selection\20201019\Result\Ch.mat', 'ch');
save('D:\Project\2020\NIRS\Channel_selection\20201019\Result\indices.mat', 'indices');
save('D:\Project\2020\NIRS\Channel_selection\20201019\Result\Inner_indice.mat', 'Inner_indice');