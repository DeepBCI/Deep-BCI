clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2020\NIRS\Channel_selection\20201102\data\op_info');
load('D:\Project\2020\NIRS\Channel_selection\20201102\data\op_loc');
load('D:\Project\2020\NIRS\Channel_selection\20201102\data\sd_loc');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};

load('D:\Project\2020\NIRS\Channel_selection\20201102\Result\indices.mat');
load('D:\Project\2020\NIRS\Channel_selection\20201102\Result\inner_indice.mat');
load('D:\Project\2020\NIRS\Channel_selection\20201102\Result\Select.mat');
load('D:\Project\2020\NIRS\Channel_selection\20201102\Result\ch.mat');

for cond = 1:4
    disp(['Condition = ' num2str(cond)])
    for sub = 1:15
        disp(['Subject = ' subdir_list{sub}])
        load(['D:\Project\2020\NIRS\Channel_selection\20201102\data\fv\fv_' subdir_list{sub} '.mat']);
        for iter = 1:5
            for fold = 1:10
                test = (indices{sub}{iter} == fold);
                train = ~test;
                for feat = 1:20
                    CH.order{cond}{sub}{iter}{fold}(feat) = find(Select{cond}{sub}{iter}{fold}(feat)==ch{sub}.num);
                    sorting(3*feat-2:3*feat,:) = fv.x(3*CH.order{cond}{sub}{iter}{fold}(feat)-2:3*CH.order{cond}{sub}{iter}{fold}(feat),:);
                end
                clear feat
                
                for feat = 1:20
                    Trset = train_shrinkageLDA_classifier(sorting(1:3*feat,train), fv.y(1,train));
                    c_est(fold,feat,sub,iter) = apply_shrinkageLDA_classifier(Trset, sorting(1:3*feat,test), fv.y(1,test));
                end
                
                clear feat sorting
            end
        end
        clear fv
    end
    Acc{cond} = mean(c_est,4);
end
save('D:\Project\2020\NIRS\Channel_selection\20201102\Result\Acc.mat','Acc');
save('D:\Project\2020\NIRS\Channel_selection\20201102\Result\CH_order.mat','CH');