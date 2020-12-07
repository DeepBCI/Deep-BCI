clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2020\NIRS\Channel_selection\20201207\data\op_info');
load('D:\Project\2020\NIRS\Channel_selection\20201207\data\op_loc');
load('D:\Project\2020\NIRS\Channel_selection\20201207\data\sd_loc');
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
[opy opx] = find(op_loc>0);
ch_num = 5;

for iter = 1:10
    clearvars -except iter op_s op_d op_info op_loc opx opy subdir_list ch_num
    
    kfold = 10;
    pseudo = [5 10 15 20 25 27];
    
    for tunit = 1:6
        for sub = 1:15
            disp(['Subject = ' subdir_list{sub}])
            load(['D:\Project\2020\NIRS\Channel_selection\20201207\data\fv\fv_' subdir_list{sub} '.mat']);
            
            ch{sub}.size = size(fv.clab,2);
            for search = 1:ch{sub}.size
                ch{sub}.num(1,search) = sscanf(cell2mat(fv.clab(search)),strcat("CH","%d","oxy"));
                [ch{sub}.cyloc(1,search), ch{sub}.cxloc(1,search)] = find(op_loc==ch{sub}.num(1,search));
                [ch{sub}.syloc(1,search), ch{sub}.sxloc(1,search)] = find(op_s == op_info(ch{sub}.num(1,search),2));
                [ch{sub}.dyloc(1,search), ch{sub}.dxloc(1,search)] = find(op_d == op_info(ch{sub}.num(1,search),3));
            end
            ch{sub}.order = 1:size(fv.clab,2);
            ch{sub}.opnum = op_info(ch{sub}.num(1,:),2:3)';
            
            MA.x = fv.x(:,logical(fv.y(1,:)));
            BL.x = fv.x(:,~fv.y(1,:));
            
            tr.x = [MA.x(:,1:pseudo(tunit)) BL.x(:,1:pseudo(tunit))];
            tr.y = [ones(1,pseudo(tunit)) zeros(1,pseudo(tunit))];
            te.x = [MA.x(:,pseudo(tunit)+1:end) BL.x(:,pseudo(tunit)+1:end)];
            te.y = [ones(1,30-pseudo(tunit)) zeros(1,30-pseudo(tunit))];
            Inner_indice{sub} = crossvalind('Kfold', pseudo(tunit)*2, kfold);
            
            %% SFS Global
            tic
            temp = ch{sub}.order;
            for channel = 1:ch_num
                if channel == 1
                    for sfs = 1:length(temp)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{sub} == Inner_k);
                            Inner_train = ~Inner_test;
                            Trset = train_shrinkageLDA_classifier(tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train), tr.y(1,Inner_train));
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test), tr.y(1,Inner_test));
                        end
                        cv_acc(sfs) = mean(c_est);
                    end
                    [~,accmax] = max(cv_acc);
                    
                    SFS(channel) = temp(accmax);
                    tempfeat = tr.x(3*temp(accmax)-2:3*temp(accmax),:);
                    temp(accmax) = [];
                    clear sfs c_est cv_acc accmax
                else
                    for sfs = 1:length(temp)
                        for Inner_k = 1:kfold
                            Inner_test = (Inner_indice{sub} == Inner_k);
                            Inner_train = ~Inner_test;
                            Trset = train_shrinkageLDA_classifier([tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], tr.y(1,Inner_train));
                            c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], tr.y(1,Inner_test));
                        end
                        cv_acc(sfs) = mean(c_est);
                        temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                        temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                        optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                        
                        loc1 = mean([ch{sub}.cxloc(SFS(:)); ch{sub}.cyloc(SFS(:))]',1);
                        loc2 = [ch{sub}.cxloc(temp(sfs)) ch{sub}.cyloc(temp(sfs))];
                        dist(sfs) = pdist([loc1; loc2]);
                        clear loc1 loc2 dist_temp temp_s temp_d
                    end
                    clear sfs c_est
                    
                    fitness = cv_acc - (optode/52) - (dist/25.02);
                    [~,accmax] = max(fitness);
                    
                    SFS(channel) = temp(accmax);
                    tempfeat = [tempfeat; tr.x(3*temp(accmax)-2:3*temp(accmax),:)];
                    temp(accmax) = [];
                    clear a b optode temp_s temp_d cv_acc accmax dist fitness
                end
            end
            Select{1}{sub}{tunit} = ch{sub}.num(SFS);
            clear channel temp SFS
            
            %% SFS_Op_Alpha
            disp('SFS_Op_Alpha')
            for alpha = 9
                tic
                disp(['Alpha = ' num2str(alpha/10)]);
                temp = ch{sub}.order;
                for channel = 1:ch_num
                    if channel == 1
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier(tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train), tr.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test), tr.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                        end
                        [~,accmax] = max(cv_acc);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = tr.x(3*temp(accmax)-2:3*temp(accmax),:);
                        temp(accmax) = [];
                        clear sfs c_est cv_acc accmax
                    else
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier([tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], tr.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], tr.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                            temp_s = [ch{sub}.opnum(1,SFS(:)) ch{sub}.opnum(1,temp(sfs))];
                            temp_d = [ch{sub}.opnum(2,SFS(:)) ch{sub}.opnum(2,temp(sfs))];
                            optode(sfs) = length(unique(temp_s))+length(unique(temp_d));
                            
                            clear loc1 loc2 dist_temp
                        end
                        clear sfs c_est
                        
                        fitness = (1-(alpha/10)) * cv_acc - (alpha/10) * (optode/52);
                        [~,accmax] = max(fitness);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = [tempfeat; tr.x(3*temp(accmax)-2:3*temp(accmax),:)];
                        temp(accmax) = [];
                        clear a b optode temp_s temp_d cv_acc accmax dist
                    end
                end
                Select{2}{sub}{tunit} = ch{sub}.num(SFS);
                toc
                clear channel
            end
            clear alpha temp SFS
            
            
            %% SFS_Dist_Beta (global)
            
            disp('SFS_Dist_Beta (global)')
            for beta = 7
                tic
                disp(['beta = ' num2str(beta/10)]);
                temp = ch{sub}.order;
                for channel = 1:ch_num
                    if channel == 1
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier(tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train), tr.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test), tr.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                        end
                        [~,accmax] = max(cv_acc);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = tr.x(3*temp(accmax)-2:3*temp(accmax),:);
                        temp(accmax) = [];
                        clear sfs c_est cv_acc accmax
                    else
                        for sfs = 1:length(temp)
                            for Inner_k = 1:kfold
                                Inner_test = (Inner_indice{sub} == Inner_k);
                                Inner_train = ~Inner_test;
                                Trset = train_shrinkageLDA_classifier([tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_train); tempfeat(:,Inner_train)], tr.y(1,Inner_train));
                                c_est(Inner_k) = apply_shrinkageLDA_classifier(Trset, [tr.x(3*temp(sfs)-2:3*temp(sfs),Inner_test); tempfeat(:,Inner_test)], tr.y(1,Inner_test));
                            end
                            cv_acc(sfs) = mean(c_est);
                            
                            loc1 = mean([ch{sub}.cxloc(SFS(:)); ch{sub}.cyloc(SFS(:))]',1);
                            loc2 = [ch{sub}.cxloc(temp(sfs)) ch{sub}.cyloc(temp(sfs))];
                            dist(sfs) = pdist([loc1; loc2]);
                            clear loc1 loc2 dist_temp
                        end
                        clear sfs c_est
                        
                        fitness = (1 - (beta/10)) * cv_acc - (beta/10) * (dist/25.02);
                        [~,accmax] = max(fitness);
                        dist_indi(channel-1) = dist(accmax);
                        
                        SFS(channel) = temp(accmax);
                        tempfeat = [tempfeat; tr.x(3*temp(accmax)-2:3*temp(accmax),:)];
                        temp(accmax) = [];
                        clear a b optode temp_s temp_d cv_acc accmax dist
                    end
                end
                clear channel
                Select{3}{sub}{tunit} = ch{sub}.num(SFS);
                toc
            end
            clear beta
            clear MA BL tr te fv temp SFS
        end
    end
    mkdir(['D:\Project\2020\NIRS\Channel_selection\20201207\Result\Online_' num2str(ch_num) 'Ch']);
    save(['D:\Project\2020\NIRS\Channel_selection\20201207\Result\Online_' num2str(ch_num) 'Ch\Select_' num2str(iter) '.mat'], 'Select');
    save(['D:\Project\2020\NIRS\Channel_selection\20201207\Result\Online_' num2str(ch_num) 'Ch\Ch_' num2str(iter) '.mat'], 'ch');
    save(['D:\Project\2020\NIRS\Channel_selection\20201207\Result\Online_' num2str(ch_num) 'Ch\Inner_indice_' num2str(iter) '.mat'], 'Inner_indice');
end