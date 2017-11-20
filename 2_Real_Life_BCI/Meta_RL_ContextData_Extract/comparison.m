clear;

list_sbj={'BMR10';'CYY03';'GBY06';'HGY13';'HJH05';'HWP08';'JGS04';'JYJ14';'JYR15';'KJA01';'KJE01';'KMJ01';'KWK11';'LYP16';'OYK09';'PCH02';'PSJ12';'YCH07';'KSY17';'OCK18'};
% list_sbj= {'KJA01','PCH02','CYY03','JGS04','HJH05','GBY06','YCH07','HWP08','OYK09','BMR10','KWK11','PSJ12','HGY13','JYJ14','JYR15','LYP16','KSY17','OCK18'};


BS = cell(1, length(list_sbj));
for subind = 1 : length(list_sbj)
    matname = [pwd '\result_save\' 'SBJ_' list_sbj{subind}  '*DPON*.mat'];
    matlist = dir(matname);
    setmodel = cell(length(matlist),3);
    for i = 1 : length(matlist)
        load([pwd '\result_save\' matlist(i).name]);
        resb = length(SBJtot);
        RESULT = cell(resb,1);
        RESULT_SB = zeros(resb,1);
        for ii = 1 : resb
            SBSBJ = SBJtot{ii,1};
            RESULT_SB(ii) = SBSBJ{1,1}.model_BayesArb.val;
            RESULT{ii,1} = SBSBJ{1,1}.model_BayesArb.param;
        end
        MINMIN = min(RESULT_SB);
        findindmin = find(RESULT_SB == MINMIN);
        OPTIMPARAM = RESULT{findindmin(1),1};
        setmodel{i,1} = SBJtot{findindmin(1),1}{1, 1}.model_BayesArb.val;
        setmodel{i,2} = SBJtot{findindmin(1),1}{1, 1}.model_BayesArb.param;
        setmodel{i,3} = SBJtot{findindmin(1),1}{1, 1}.num_data;
    end
    BS{1,subind} = setmodel;
end

resbs = length(BS);
valset = cell(resbs,1);
paramset = cell(resbs,1);
minRef = 99999;
SBJset = cell(resbs,1);
for i = 1 : resbs
    ind = size(BS{1,i},1);
    valset{i,1} = minRef;
    for in = 1 : ind
        mintemp = BS{1,i}{in,1};
        if mintemp < valset{i,1}
            valset{i,1} = mintemp;
            indexs = in;
        end
    end
    paramset{i,1} = BS{1,i}{indexs,2};
    SBJset{i} = BS{1,i}{indexs,3};
end
paramset = cell2mat(paramset);

%% Model comparison
% load for both comparison