function [SBJ]=ArbBat1(maxi,mode, PE, MODELS,PreBehav, PreBlck,MainBehav,MainBlck,list_sbj,param_init)
tt = clock;
rng(floor(tt(6)*1000));
for ppi = 1 : 3
    rng(floor(tt(6)*1000));
    param_init(ppi) = randi(20)*(mode.param_BoundU(ppi) - mode.param_BoundL(ppi))/20  + mode.param_BoundL(ppi);
end
mode.param_init_ori = param_init;
for i = 1 : maxi
    exp0=clock;
    fprintf('###   SUB_NUM: [%d / %d]\n',i,maxi);
    fprintf('### OPT_ITER : [%d]\n',mode.max_iter);
    disp('############################################');
    disp('############################################');
    disp('############################################');
    % pre save
    data_in{1,1}.map_type=1;
    data_pre=load([pwd '/result_save/' list_sbj{i} '_pre_1.mat']);
    data_in{1,1}.HIST_behavior_info_pre{1,1}=PreBehav{i};
    data_in{1,1}.HIST_block_condition_pre{1,1}=PreBlck{i};
    % data MODELS/ PE save
    data_in{1,1}.DPNMM_MODEL = MODELS{1,i};
    data_in{1,1}.RPE_sarsa = PE{1,i}.RPE_SARSA;
    data_in{1,1}.SPE_T = PE{1,i}.SPE_T;
    % max sess eval
    % main save
    temp = [];
    t2t = dir([pwd '/result_save']);
    t2t = {t2t.name};
    maxsess = sum(cell2mat(strfind(t2t,[list_sbj{i} '_fmri_']))) - 1;
    for ii = 1 : maxsess
        data_in{1,1}.HIST_behavior_info{1,ii} = MainBehav{i,ii};
        data_in{1,1}.HIST_block_condition{1,ii} = MainBlck{i,ii};
    end
    % DPNMM save
    % mode.DPNMM = DPNMMset{i};
    
    
    %         % NO opt process
    %         outvalval = eval_ArbitrationRL6c(param_init, data_in, mode);
    %         outputval{d}=[outputval{d} outvalval];
    
    
    % optimization part
    myFunc_bu = @(x) eval_ArbitrationRL_DPON2(x, data_in, mode);
    disp(['    ***** subject number : [' num2str(i) '], subject name : [' list_sbj{i} ']' ]);
    einstein= 1;
    howmany=0;
    
    while (einstein==1)
        try
            
            [model_BayesArb.param, model_BayesArb.val] = fminsearchbnd(myFunc_bu, param_init, mode.param_BoundL, mode.param_BoundU, optimset('Display', 'iter','MaxIter',mode.max_iter)); % X0,LB,UB
            einstein= 0;
            clc;
        catch
            einstein=1;
            rng(floor(tt(6)*1000));
            for ppi = 1 : 3
                rng(floor(tt(6)*1000));
                param_init(ppi) = randi(20)*(mode.param_BoundU(ppi) - mode.param_BoundL(ppi))/20  + mode.param_BoundL(ppi);
            end
            mode.param_init_ori = param_init;
            howmany= howmany + 1;
            disp([num2str(howmany) 'error(s)']);
            if howmany > 1000
                einstein=0;
            end
        end
    end
    model_BayesArb.mode = mode;
    % for Storing
    SBJ{1,i} = data_in{1,1};
    SBJ{1,i}.model_BayesArb = model_BayesArb;
    SBJ{1,i}.num_data = PE{1,i}.trials;    
    disp('############################################');
    disp('############################################');
    disp('############################################');
end
end