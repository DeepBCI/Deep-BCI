function batch_parameter_recovery_behaviors()



betas_colike_wo_act = [];
betas_colike_wo_act_inter = [];
betas_colike_gc2_w_sta2_inter =[];
betas_colike_gc2_w_sta2 = [];
betas_colike_w_sta2_inter = [];
betas_colike_w_sta2 = [];
for i = 1 : 1 : 10
    
    T = readtable(['bhv_results/' sprintf('SUB%03d_BHV.csv',i)]);
    dat= table2array(T);
    act = dat(:,7);
    glc = dat(:,18);
    unc = ones(size(glc));
    unc ( union(find(dat(:,3) == 2), find(dat(:,3) == 3)) ) = 2;
    % 1: specific 2: flexible
    gc2 = ~logical(double(dat(:,18)<0)) +1 ;
    % 3: specific, 2 spec-flex 1: flexible
    gc3 = double(dat(:,18)<0)*2 +1 ;
    gc3(find(dat(:,18)==6)) = 2;
    
    res_sta1=[];
    res_sta4=[];
    res_stage2=[];
    token_value = [0 0 0 0 0 40 20 10 0];
    if dat(1,18) == -1
        nr = [dat(1,16)/40 dat(1,3)];
    else
        nr = [dat(1,16)/token_value(dat(1,18)) dat(1,3)];
    end
    [~ , opti0] = opt_normalization_1st2ndstage_exnan(1, dat(1,7), dat(1,18), dat(1,3)); 
    prev_opti1 = opti0;
    
    prev_action_sta1 = dat(1,7);
    if dat(1,5)==4
        prev_action_sta4 = dat(1,8);
        [~, opti04] = opt_normalization_1st2ndstage_exnan(dat(1,5), dat(1,8), dat(1,18), dat(1,3)); 
        prev_opti4 = opti04;

        prev4on = 1;
    else
        prev4on = 0;
    end
    prev_action_stage2 = dat(1,8);
    [~, opti02] = opt_normalization_1st2ndstage_exnan(dat(1,5), dat(1,8), dat(1,18), dat(1,3));
    prev_opti2 = opti02;

    for k = 2 : 1: length(dat)
        curr_action_sta1 = dat(k,7);
        curr_action_stage2 = dat(k,8);
        
        if dat(k,18) == -1
            nr = [nr; [dat(k,16)/40 dat(k,3)]];
        else
            nr = [nr; [dat(k,16)/token_value(dat(k,18)) dat(k,3)]];
        end
       
        % choice consistency/ choice switch/ choice optimality / switch to
        % optiamal choices / context / pmb / motiv mb/ motiv mf
        % sta1
        [opti_likeli1, opti1] = opt_normalization_1st2ndstage_exnan(1, dat(k,7), dat(k,18), dat(k,3)); 
        curr_opti1 = opti1;
        res_sta1=[res_sta1; [curr_action_sta1==prev_action_sta1 curr_action_sta1~=prev_action_sta1 opti_likeli1 (curr_action_sta1~=prev_action_sta1)&&(ismember(curr_action_sta1,opti1)) dat(k,3) 0 0 0 ismember(curr_action_sta1,opti1)]];
        
        % sta4
        if dat(k,5) == 4
            if prev4on == 0
                prev_action_sta4 = dat(k,8);
                [~, opti04] = opt_normalization_1st2ndstage_exnan(dat(k,5), dat(k,8), dat(k,18), dat(k,3)); 
                prev_opti4 = opti04;
                prev4on = 1;
            else
                prev4on = 1;
                curr_action_sta4 = dat(k,8);
                [opti_likeli4, opti4] = opt_normalization_1st2ndstage_exnan(dat(k,5), dat(k,8), dat(k,18), dat(k,3)); 
                curr_opti4 = opti4;
                res_sta4=[res_sta4; [curr_action_sta4==prev_action_sta4 curr_action_sta4~=prev_action_sta4 opti_likeli4 (curr_action_sta4~=prev_action_sta4)&&(ismember(curr_action_sta4,opti4)) dat(k,3) 0 0 0 ismember(curr_action_sta4,opti4)]];
                
                prev_action_sta4 = dat(k,8);
                prev_opti4 = opti4;
            end
        end
        
        % stage2
        [opti_likeli2, opti2] = opt_normalization_1st2ndstage_exnan(dat(k,5), dat(k,8), dat(k,18), dat(k,3)); 
        curr_opti2 = opti2;
        res_stage2=[res_stage2; [curr_action_stage2==prev_action_stage2 curr_action_stage2~=prev_action_stage2 opti_likeli2 (curr_action_stage2~=prev_action_stage2)&&(ismember(curr_action_stage2,opti2)) dat(k,3) 0 0 0 ismember(curr_action_stage2,opti2)]];

        
        prev_opti1 = curr_opti1;
        prev_opti2 = curr_opti2;
        prev_action_sta1 = dat(k,7);
        prev_action_stage2 = dat(k,8);
    end
    
    %%%%%%%%%%%%%%
%     res = res_sta1;
    res = res_stage2;
    
    ds2 = dataset(unc(2:end),gc3(2:end),res(:,3),'Varnames',{'Uncertainty','GoalCondition','ChoiceOptimality(likeli)'});

    mdl_2 = fitglm(ds2,'interactions');
    mdl2_2 = fitglm(ds2,'linear');
    B_2 = mdl_2.Coefficients.Estimate;
    B2_2 = mdl2_2.Coefficients.Estimate;
    
    betas_colike_wo_act_inter = [betas_colike_wo_act_inter;B_2'];
    betas_colike_wo_act = [betas_colike_wo_act;B2_2'];
    
    ds3 = dataset(unc(2:end),gc2(2:end),act(1:end-1),dat(2:end,6),res(:,3),'Varnames',{'Uncertainty','GoalCondition','PrevAction','sta2','ChoiceOptimality(likeli)'});
    ds4 = dataset(unc(2:end),gc3(2:end),act(1:end-1),dat(2:end,6),res(:,3),'Varnames',{'Uncertainty','GoalCondition','PrevAction','sta2','ChoiceOptimality(likeli)'});
    
    mdl = fitglm(ds3,'interactions');
    mdl2 = fitglm(ds3,'linear');
    B = mdl.Coefficients.Estimate;
    B2 = mdl2.Coefficients.Estimate;
    betas_colike_gc2_w_sta2_inter = [betas_colike_gc2_w_sta2_inter; B'];
    betas_colike_gc2_w_sta2 = [betas_colike_gc2_w_sta2; B2'];
    
    mdl_2 = fitglm(ds4,'interactions');
    mdl2_2 = fitglm(ds4,'linear');
    B_2 = mdl_2.Coefficients.Estimate;
    B2_2 = mdl2_2.Coefficients.Estimate;    
    betas_colike_w_sta2_inter = [betas_colike_w_sta2_inter; B_2'];
    betas_colike_w_sta2 = [betas_colike_w_sta2; B2_2'];
end

save(['betas_colike_wo_act_inter_BEHAV.mat'],'betas_colike_w_sta2_inter');
% save(['betas_colike_wo_act_BEHAV.mat'],'betas_colike_wo_act');
