clear all
close all
clc

startup_bbci_toolbox('DataDir','D:\Project\2019\NIRS\Channel_selection\data', 'TmpDir','/tmp/');
load('D:\Project\2021\NIRS\data\Discription');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};
num_channel = 20;

for sub = 1:15
    disp(['Subject = ' subdir_list{sub}])
    load(['D:\Project\2022\NIRS_selection\Result\0110\Selected_' num2str(sub) '.mat'])
    load(['D:\Project\2021\NIRS\data\fv\fv_' subdir_list{sub} '.mat']);
    for alpha = 1:11
        for fold = 1:10
            test = (indices == fold);
            train = ~test;
            for feat = 1:num_channel
                a = find(Select{alpha,fold}(feat)==ch.num);
                CH.order{alpha,sub,fold}(feat) = a(1);
                sorting(3*feat-2:3*feat,:) = fv.x(3*CH.order{alpha,sub,fold}(feat)-2:3*CH.order{alpha,sub,fold}(feat),:);
            end
            clear feat
            
            for feat = 1:num_channel
                Trset = train_shrinkageLDA_classifier(sorting(1:3*feat,train), fv.y(1,train));
                c_est(feat,fold) = apply_shrinkageLDA_classifier(Trset, sorting(1:3*feat,test), fv.y(1,test));
            end
            clear feat sorting
        end
        Acc(12-alpha,:,sub) = mean(c_est,2);
    end
end
save('D:\Project\2022\NIRS_selection\Result\0110\Acc.mat','Acc');
save('D:\Project\2022\NIRS_selection\Result\0110\CH_order.mat','CH');