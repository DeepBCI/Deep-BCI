function [PE,MODELS] = PEb2(maxi,list_sbj,PreBehav,PreBlck,MainBehav,MainBlck,param_init,mode,t)
list_month={'Jan','Feb','Mar','Apr', 'May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec'};

for i = 1 : maxi
    tt = dir([pwd '/result_save']);
    tt = {tt.name};
    maxsess = sum(cell2mat(strfind(tt,[list_sbj{i} '_fmri_']))) - 1;
    % 임시로 data_in 구성하는 부분좀 가져오자.
    
    DATA_IN{1,1}.HIST_behavior_info_pre{1,1}=PreBehav{i};
    DATA_IN{1,1}.HIST_block_condition_pre{1,1}=PreBlck{i};
    for ii = 1 : maxsess
        DATA_IN{1,1}.HIST_behavior_info{1,ii} = MainBehav{i,ii};
        DATA_IN{1,1}.HIST_block_condition{1,ii} = MainBlck{i,ii};
    end
    PE{1,i} = Init_PE(DATA_IN,param_init,mode); %
end
mkdir([ pwd '/result_save/' num2str(t(1)) list_month{t(2)} num2str(t(3))]);
save([ pwd '/result_save/' num2str(t(1)) list_month{t(2)} num2str(t(3)) '/F_PE_Simul_' num2str(i) '.mat'], 'PE');
disp('PE STORING DONE!');

alpha=1;
MODELS={};
for i = 1 : maxi
    MODELS{1,i}=Init_DirichletProcessNormalMix_Arbitration_Init(PE{1,i},alpha);
end
save([ pwd '/result_save/' num2str(t(1)) list_month{t(2)} num2str(t(3)) '/F_DPNMM_MODEL' num2str(alpha) '_SIMUL_' num2str(i)  '_Oracle.mat'], 'MODELS');
disp('MODELS STORING DONE!');

end