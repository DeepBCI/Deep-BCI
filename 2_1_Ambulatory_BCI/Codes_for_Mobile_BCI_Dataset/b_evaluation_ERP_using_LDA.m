%%
ival_cfy_fixing = 1; % 1:ture // fixing

AUC_all = [];

for subNum=1:nSub
%% select channels
% chan = {'C3','C1','C2' ,'C4','CP1','CP2','P3','Pz','P4','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'}; % scalp-EEG
chan = {'L1','L2','L4','L5','L6','L7','L9','L10','R1','R2','R4','R5','R7','R8'}; % ear-EEG

%% ival setting

ref_ival= [-200 0] ;
r_ival = [100 600];

ival_cfy = [200 250; 250 300; 300 350; 350 400; 400 450];

%% training
epo = EPO{subNum,1};

epo = proc_selectChannels(epo, chan);

% select ival
if ival_cfy_fixing == false
    epo_r= proc_selectIval(epo, r_ival, 'IvalPolicy','minimal');
    epo_r= proc_rSquareSigned(epo_r);
    ival_cfy= procutil_selectTimeIntervals(epo_r);
    disp('ival cfy non-fixing  -  Check ival_cfy')
end

fv_Tr= proc_baseline(epo, ref_ival);
fv_Tr= proc_jumpingMeans(fv_Tr, ival_cfy);

xsz= size(fv_Tr.x);
fvsz= [prod(xsz(1:end-1)) xsz(end)];

% classifier C train_RLDAshrink
C = train_RLDAshrink(reshape(fv_Tr.x,fvsz), fv_Tr.y);

for speedIdx = 2:sum(~cellfun('isempty', EPO(subNum,:)))
%% test
epo = EPO{subNum,speedIdx};

epo = proc_selectChannels(epo, chan);

fv_Te= proc_baseline(epo, ref_ival);
fv_Te= proc_jumpingMeans(fv_Te, ival_cfy);

xTesz= size(fv_Te.x);

% test loss
outTe= apply_separatingHyperplane(C, reshape(fv_Te.x, [prod(xTesz(1:end-1)) xTesz(end)]));
lossTe = mean(loss_0_1(fv_Te.y, outTe));

% training loss
outTr= apply_separatingHyperplane(C, reshape(fv_Tr.x, fvsz));
lossTr = mean(loss_0_1(fv_Tr.y, outTr));

% Get AUC
[ERP_per.roc, ERP_per.auc]= roc_curve(epo.y, outTe,'plot',0);
AUC_all(speedIdx-1, subNum) = ERP_per.auc;

end
end

%% Average AUC per Speed
disp('Mean AUC')
for ispeed = 1:4
mean_AUC(ispeed,1) = sum(AUC_all(ispeed,:))/nnz(AUC_all(ispeed,:));
end
disp(mean_AUC)
