
function [out]=SIMUL_regressor_v1()
warning('off');

% For saving figures, 'savefigs'

path0=pwd;
seed_path_result=[path0];
save_path_result=[path0];
save_for_SPM=[pwd '/uncertainty_arbitration/regressors_contrasts/'];
save_path_neuroecon= save_for_SPM;


% 1. Behavioral data
% LIST_SBJ={'david', 'DeDe', 'rosemary', 'Boyu', 'melissa', 'Rehevolew', 'joel', 'clarke', 'angela', 'william', 'josephine'}; % (good in pre which is mostly habitual - rosemary, melissa)
% mode.map_type=?;

% 2. behavioral + fmri data (for map config, see SIMUL_arbitraion_fmri2.m)
% [note] 'Oliver' uses an old map. the rest of them use a new map.


LIST_sbj_map_type=[ 2*ones(1,41)]; %1:'sangwan2012b', 2:'sangwan2012c'


LIST_SBJ={'BMR10';'CYY03';'GBY06';'HGY13';'HJH05';'HWP08';'JGS04';'JYJ14';'JYR15';'KJA01';'KJE01';'KMJ01';'KWK11';'LYP16';'OYK09';'PCH02';'PSJ12';'YCH07';'KSY17';'OCK18'};
% LIST_SBJ = {'KJA01' 'PCH02' 'CYY03' 'JGS04' 'HJH05' 'GBY06' 'YCH07' 'HWP08' 'OYK09' 'BMR10' 'KWK11' 'PSJ12' 'HGY13' 'JYJ14' 'JYR15' 'LYP16' 'KSY17' 'OCK18'};

% regressor list
% [CAUTION] DO NOT change the order!!!
% [NOTE] if "TYPE_REGRESSOR" changed, change "param_regressor_type_cue_abs_pos_in_design_mat" accordingly!!!
LIST_REGRESSOR={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'meanM1', 'meanM2', 'invFanoM1', 'invFanoM2', 'weigtM1', 'weigtM2', 'Qfwd', 'Qsarsa', 'Qarb', 'dQbwdEnergy', 'dQbwdMean', 'duncertaintyM1', 'dinvFanoM1'};
TYPE_REGRESSOR=[1 1, 1.5 1.5, 1.5 1.5, 1.5 1.5, 2 2, 1.5 1.5 1.5 1.5 1.5 1.5 1.5]; % 1: parametric modulation (0-duration), 1.5:parmetric modulation (non-zero duration), 1.7:parametric modulation (with decision onset)  2: extra continuous parametric modulation (TR-fixed) - this will be used by "dummy" regressor.
row_mat=[7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7]; % from which row in the SBJ{}.regressor matrix the signal needs to be extracted. e.g., uncertainty of 0 prediction error

%% OPTION - subject
% [note] DO NOT USE sbj#[20] - he pressed wrong buttons in session1,2, so need to shrink all the SBJ matrix size by deleting the session#1,2
list_sbj_included=[1:1:length(LIST_SBJ)];

%% OPTION - model optimization
option_optimizing_model=0; % 0: optimizing the model for each sbj, 1: for all sbj, 2: do not optimize; load saved model
update_SBJ_structure=0; % 0: no update/just read and use, 1: update the changes to the saved SBJ file
mode.USE_FWDSARSA_ONLY=0; % 0: arbitration, 1: use fwd only, 2: use sarsa only
mode.USE_BWDupdate_of_FWDmodel=1; % 1: use the backward update for goal-directed model (fwd model), 0: do not use
mode.DEBUG_Q_VALUE_CHG=0; % Debug option 1: show Q-value before/after whenever there is a goal change.
mode.path_ext=path0;
mode.total_simul=20; % # of total simulation repetition per subject
mode.simul_process_display=0; % 1: display model's process, 0: no diplay
mode.experience_sbj_events=[+1 1]; % [pre main]  +1: experience exactly the same events(decision,state) as subjects. 0: model's own experience -1: use saved setting
mode.max_iter=100; % maximum iteration for optimization
% mode.out=1; % 1: normal evaluation mode, 99: regressor added to the SBJ, 0: debug mode
mode.param_length = 4 ;%size(param_init,2);
mode.opt_ArbModel = 1; % % 1: naive model(m1_wgt) . 2: posterior model(posterior)
mode.boundary_12 = 0.1; % 위아래거 바뀔수도 있음 사실 뭐가 맞는지 모르겟음...
mode.boundary_21 = 0.01;




%% OPTION - Regressor arrangement
% {'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd', 'Qarb','uncertaintyM2', 'invFanoM2', 'duncertaintyM1', 'dinvFanoM1', 'weigtM1'};
% should add the regressors in the order of importance
param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd','Qarb','uncertaintyM2', 'invFanoM2', 'weigtM1'};
reg_type_go_first=[1 1.5 2]; % [CAUTION] The order should match with 'param_regressor_type_cue'.   [CAUTION] type"2" should go always last!!!
Do_create_regressors=1;
Is_save_files_local=1; % save optimization parameters and regressor files
Is_save_files_cluster=0; % save optimization parameters and regressor files


%% OPTION - behaviroal analysis & display
Do_behavioral_analysis=[0];
if(Do_behavioral_analysis(1)==1) % dont need to create regressors in behavioral analysis mode!
    Do_create_regressors=0;
end

%% initialization
if(Is_save_files_local==0)    disp('### files will not be saved to your local PC.');     end
if(Is_save_files_cluster==0)    disp('### files will not be saved to the cluster PC.');     end

use_model_regressor_cue=0;  ind_regressor_total=[];   type_regressor=[];    ind_regressor_total_in_design_mat=[];
for ii=1:1:size(param_regressor_type_cue,2) % collect regressor information
    if(strcmp(param_regressor_type_cue{1,ii},'SPE')==1)    use_model_regressor_cue=1;  ind_chk=1;   end
    if(strcmp(param_regressor_type_cue{1,ii},'RPE')==1)    use_model_regressor_cue=1;  ind_chk=2;   end
    if(strcmp(param_regressor_type_cue{1,ii},'uncertaintyM1')==1)    use_model_regressor_cue=1;    ind_chk=3;   end
    if(strcmp(param_regressor_type_cue{1,ii},'uncertaintyM2')==1)    use_model_regressor_cue=1;    ind_chk=4;   end
    if(strcmp(param_regressor_type_cue{1,ii},'meanM1')==1)    use_model_regressor_cue=1;    ind_chk=5;   end
    if(strcmp(param_regressor_type_cue{1,ii},'meanM2')==1)    use_model_regressor_cue=1;    ind_chk=6;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invFanoM1')==1)    use_model_regressor_cue=1;    ind_chk=7;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invFanoM2')==1)    use_model_regressor_cue=1;    ind_chk=8;   end
    if(strcmp(param_regressor_type_cue{1,ii},'weigtM1')==1)    use_model_regressor_cue=1;    ind_chk=9;   end
    if(strcmp(param_regressor_type_cue{1,ii},'weigtM2')==1)    use_model_regressor_cue=1;    ind_chk=10;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qfwd')==1)    use_model_regressor_cue=1;    ind_chk=11;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qsarsa')==1)    use_model_regressor_cue=1;    ind_chk=12;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qarb')==1)    use_model_regressor_cue=1;    ind_chk=13;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dQbwdEnergy')==1)    use_model_regressor_cue=1;    ind_chk=14;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dQbwdMean')==1)    use_model_regressor_cue=1;    ind_chk=15;   end
    if(strcmp(param_regressor_type_cue{1,ii},'duncertaintyM1')==1)    use_model_regressor_cue=1;    ind_chk=16;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dinvFanoM1')==1)    use_model_regressor_cue=1;    ind_chk=17;   end
    
    % index of regressor in "SBJ" structure
    ind_regressor_total=[ind_regressor_total ind_chk];
    % regressor type
    type_regressor=[type_regressor TYPE_REGRESSOR(ind_chk)];
end
% make a regressor index matrix for 1st parametric modulations (normal)
% (1) make 'param_regressor_type_cue_abs_pos_in_design_mat'
reg_cnt=0; param_regressor_type_cue_abs_pos_in_design_mat=[];
ind_regressor_type_base{1,1}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(1)));
reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,1}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [2:1:reg_cnt]];
ind_regressor_type_base{1,2}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(2)));
reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,2}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+2:1:reg_cnt]];
if(reg_type_go_first(3)==1.7)
    ind_regressor_type_base{1,3}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(3)));
    reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,3}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+2:1:reg_cnt]];
end

% make a regressor index for 2nd parametric modulations (dummy)
ind_regressor_type_dummy.ind_reg=ind_regressor_total(find(type_regressor==2));
reg_cnt=reg_cnt+length(ind_regressor_type_dummy.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+1:1:reg_cnt]];
for j=1:1:length(ind_regressor_type_dummy.ind_reg)
    ind_regressor_type_dummy.name{1,j}=LIST_REGRESSOR{1,ind_regressor_type_dummy.ind_reg(j)};
end
% (2)
ind_regressor_type_base{1,1}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(1)));
ind_regressor_type_base{1,2}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(2)));
if(reg_type_go_first(3)==1.7)
    ind_regressor_type_base{1,3}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(3)));
end
ind_regressor_type_dummy.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==2));


if(sum(abs(param_regressor_type_cue_abs_pos_in_design_mat-sort(param_regressor_type_cue_abs_pos_in_design_mat,'ascend')))~=0)
    error('- ERROR!!!!: the variable ''param_regressor_type_cue_abs_pos_in_design_mat'' should be in ascending order!!!');
end



%% subject data loading
% which subject to be included
% ### READ ONLY ONE SBJ BECAUSE EACH MODEL WILL LEARN EACH SBJ BEHAVIOR.
ind_sbj_included=list_sbj_included;      SUB_ARRAY=list_sbj_included;
num_sbj_included=length(ind_sbj_included);
ind_included=ind_sbj_included;

for k=1:1:num_sbj_included
    LIST_SBJ_included{1,k}=LIST_SBJ{ind_sbj_included(k)};
end
for i=1:1:num_sbj_included %=1. process only 1 subject
    
    SBJ{1,i}.name=LIST_SBJ{ind_sbj_included(i)};
    
    % 'pre' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=['_pre_info.mat'];
    file_name_full=[mode.path_ext '\result_save\' LIST_SBJ{ind_sbj_included(i)} file_name];
    filefile = dir(file_name_full);
    load([mode.path_ext '\result_save\' filefile.name]);
    SBJ{1,i}.HIST_block_condition_pre=HIST_block_condition;
    SBJ{1,i}.HIST_behavior_info_pre=HIST_behavior_info;
    
    % 'fmri' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=['_fmri_info.mat'];
    file_name_full=[mode.path_ext '\result_save\' LIST_SBJ{ind_sbj_included(i)} file_name];
    filefile = dir(file_name_full);
    load([mode.path_ext '\result_save\' filefile.name]);
    SBJ{1,i}.HIST_behavior_info=HIST_behavior_info;
    SBJ{1,i}.HIST_behavior_info_Tag=HIST_behavior_info_Tag;
    SBJ{1,i}.HIST_event_info=HIST_event_info;
    SBJ{1,i}.HIST_event_info_Tag=HIST_event_info_Tag;
    SBJ{1,i}.HIST_block_condition=HIST_block_condition;
    SBJ{1,i}.HIST_block_condition_Tag=HIST_block_condition_Tag;
    num_tot_session=size(SBJ{1,i}.HIST_behavior_info,2);
    
    SBJ{1,i}.map_type=LIST_sbj_map_type(ind_sbj_included(i));
    
    % [fixing part!!! - for Oliver]
    if(strcmp(SBJ{1,i}.name,'Oliver'))
        for mm=1:1:size(SBJ{1,i}.HIST_event_info,2) % each session
            mat_fixing=SBJ{1,i}.HIST_event_info{1,mm};
            index_delete=zeros(1,size(mat_fixing,2));
            [r_fix, c_fix]= find(mat_fixing(7,:)==9);
            for nn=1:1:length(c_fix)
                % check the previous event
                if(mat_fixing(7, c_fix(nn)-1)~=0.5)
                    index_delete(c_fix(nn))=1;
                end
            end
            [tmp c_keep]=find(index_delete==0);
            mat_fixed=mat_fixing(:,c_keep);
            SBJ{1,i}.HIST_event_info{1,mm}=mat_fixed;
        end
    end
    
    
    % [NOTE] now we have 4 variables: mode.HIST_block_condition_pre, mode.HIST_block_condition, mode.HIST_behavior_info_pre, mode.HIST_behavior_info
    % to read a block condition, use "block_condition=mode.HIST_block_condition{1,session_ind}(2,block_ind); % G:1,G':2,H:3,H':4"
    
    %     swsw_amount_pre = [swsw_amount_pre mode.HIST_behavior_info_pre{1,1}(end,17)];
    tot_amount_earned_main_each_sbj =[];
    for jk=1:1:size(SBJ{1,i}.HIST_behavior_info,2)    tot_amount_earned_main_each_sbj = [tot_amount_earned_main_each_sbj; SBJ{1,i}.HIST_behavior_info{1,jk}(end,17)]; end
    %     swsw_amount_main=[swsw_amount_main tot_amount_earned_main_each_sbj];
end







%% model optimization loading load from already optimized data set.
% loading best set of parameters [2017.03.08 , KDJ]
optim_param = 'bestSBJ*.mat';% optimal parameter mat file prefix.
seed_path_result;
% If there is no Best Parameter Set
option_best_param_load = 0;

if option_best_param_load == 1
    % save for optimization
    list_p = dir([save_path_result optim_param]);
    SBJ2 = load([save_path_result optim_param], optim_param);
else % load from actual best param set.
    SBJ2 = cell(1,num_sbj_included);
    for i = 1:1:num_sbj_included
        SBJ_list = dir([seed_path_result '\result_save\SBJ_' LIST_SBJ_included{i} '_DPON_*']);
        sim_tot = length(SBJ_list);
        minmin=99999; % meaningless large number
        for is = 1 : 1 : sim_tot
            sbsb = load([seed_path_result '/result_save\' SBJ_list(is).name]);
            for simin = 1 : 1 : length(sbsb.SBJtot)
                if minmin > sbsb.SBJtot{simin}{1}.model_BayesArb.val;
                    minmin = sbsb.SBJtot{simin}{1}.model_BayesArb.val;
                    SBJ2{1,i} = sbsb.SBJtot{simin}{1};
                end
            end
        end
        SBJ2{1,i}.name = LIST_SBJ_included{i};
        
    end
end


% mode.param_BoundL = [0.5 0.5 0.01 0.01];
% mode.param_BoundU = [10 10 0.5 0.3];
% param_init= zeros(1,4);
% for i = 1 : 6
%     rand(floor(mod(sum(clock*10),10000)));
%     param_init(i) = rand  * (mode.param_BoundU(i) - mode.param_BoundL(i))  + mode.param_BoundL(i);
% end

% ## (way1-each) optimizing for *each* subject and plug the result into each SBJ structure
if(option_optimizing_model==0)
    for ind_sbj=1:1:size(SBJ2,2)
        clear SBJ_test;
        SBJ_test{1,1}=SBJ2{1,ind_sbj};
        disp('############################################')
        disp(['#### optimizing RL-arbitrator for ' sprintf('SBJ#%02d...',ind_sbj)]);
        disp('############################################')
        % [1] model optimization
        %         mode.out=1;
        %         myFunc_bu = @(x) eval_ArbitrationRL_DPON2(x, SBJ_test, mode); % define a new anonymous function
        %         [model_BayesArb.param, model_BayesArb.val]=fminsearchbnd(myFunc_bu, param_init, mode.param_BoundL, mode.param_BoundU, optimset('Display','iter','MaxIter',mode.max_iter));   % X0,LB,UB
        % [2-1] add regressor vector to SBJ
        mode.out=99;
        SBJ_test=eval_ArbitrationRL_DPON2(SBJ_test{1, 1}.model_BayesArb.param,SBJ_test,mode);
        % [3] Save
        model_BayesArb.mode=mode;
%         SBJ_test{1,1}.model_BayesArb=model_BayesArb;
        SBJ2{1,ind_sbj}=SBJ_test{1,1};
        save_file_name=['SBJ_structure.mat'];
        if(Is_save_files_local==1)
            eval(['save ' save_path_result '\' save_file_name ' SBJ2'])
        end
    end
    option_optimizing_model=2; % and then write regressors to SBJ structure based on this optimized parameter
end
save_file_name=['SBJ_structure.mat'];

save([save_path_result '\' save_file_name], 'SBJ2');

% if(option_optimizing_model==1)
%     % ## (way2-batch) optimizing for *all* subjects and plug the result into each SBJ structure
%     % [0] retrieve intial configuration for skipping pre-training
% %     SBJ_keep=SBJ;
% %     load_file_name=['SBJ_structure(backup,batch,Oct30_4).mat'];
% %     eval(['load ' save_path_result load_file_name]);
% %     for ff1=1:1:length(SBJ_keep)
% %         SBJ_keep{1,ff1}.init_state_fwd=SBJ{1,ff1}.init_state_fwd;    SBJ_keep{1,ff1}.init_state_sarsa=SBJ{1,ff1}.init_state_sarsa;
% %     end
% %     SBJ=SBJ_keep;
%     % [1] model optimization
%     mode.out=1;
%     myFunc_bu = @(x) eval_ArbitrationRL2(x, SBJ, mode); % define a new anonymous function
%     [model_BayesArb.param, model_BayesArb.val]=fminsearchbnd(myFunc_bu, param_init, param_BoundL, param_BoundU, optimset('Display','iter','MaxIter',mode.max_iter));   % X0,LB,UB
%     % [2-1] add regressor vector to SBJ
%     mode.out=99;
%     SBJ=eval_ArbitrationRL2(model_BayesArb.param,SBJ,mode);
%     % [3] save
%     model_BayesArb.mode=mode;
%     for ind_sbj=1:1:size(SBJ,2) % plug in a identical parameter (because this is batch)
%         SBJ{1,ind_sbj}.model_BayesArb=model_BayesArb;
%     end
%     save_file_name=['SBJ_structure.mat'];
%     if(Is_save_files_local==1)
%         eval(['save ' save_path_result save_file_name ' SBJ'])
%     end
%     option_optimizing_model=2; % and then write regressors to SBJ structure based on this optimized parameter
% end
%
% if(option_optimizing_model==2)
%     load_file_name=['SBJ_structure.mat'];
%     eval(['load ' save_path_result load_file_name])
%     % regressor part deleting and regenerating.
%     for ff=1:1:length(list_sbj_included)
%         disp(sprintf('- writing regressor to SBJ structure (SBJ%02d)...',list_sbj_included(ff)));
%         SBJ0{1,1}=SBJ{1,ff};
%         if(isfield(SBJ0{1,1}, 'regressor')==1)
%             SBJ0{1,1}=rmfield(SBJ0{1,1},'regressor'); %remove the regressor field
%         end
%         mode.out=99;
%         model_BayesArb.param=SBJ0{1,1}.model_BayesArb.param;
%         SBJ0=eval_ArbitrationRL2(model_BayesArb.param,SBJ0,mode); % refresh and add the regressor part
%         SBJ1{1,ff}=SBJ0{1,1};
%     end
%     SBJ=SBJ1;
%     if(update_SBJ_structure==1)        eval(['save ' save_path_result load_file_name ' SBJ']);  end
% end
%
%
%
% %test mode
% if(option_optimizing_model==3)
%     model_BayesArb.param=param_init;
%     mode.out=99;
%     SBJ=eval_ArbitrationRL2(model_BayesArb.param,SBJ,mode);
% end



% Must be fixed
if(Do_behavioral_analysis(1)==1)
    
    %% Behavioral analysis #1 (computational)
    
    disp_opt.sbj_ind_for_goal_directed_trace=1; % pick only one subject to display as an example
    
    all_y_goal_amount_in_goalblock=[];  all_y_goal_amount_in_habitualblock=[];
    all_out_corr=[];
    SBJ_test.HIST_event_info = SBJ.HIST_event_info;
    SBJ = SBJ_test;
    for jj2=1:1:size(SBJ,2)        % each subject
        
        %% 0. degree of goal-directed (Likelihood-ratio) "how much subjects are goal-directed?"
        % we can use a likelihood ratio as an index, but we cannot do the
        % likelihood ratio test because fwd and sarsa model are different
        % model (i.e., one is not the special case of the other)
        x_mat=[1:1:length(SBJ{1,jj2}.model_error{1,1}.value(7,:))];
        % Test statistics of likelihood-ratio test
        y_mat=2*[SBJ{1,jj2}.model_error{1,2}.value(7,:)-SBJ{1,jj2}.model_error{1,1}.value(7,:)]; % how much goal-directed (0:neutral) likelihood ratio
        y_mat_disp=y_mat/max(abs(y_mat)); %[-1,1]  for display
        % 1: goal block, -1: habitual block
        y_block_condi=(SBJ{1,jj2}.model_error{1,2}.value(5,:)~=-1)-0.5; %[-.5,.5]
        y_block_condi_goal=(y_block_condi>0); %[1,0]
        y_block_condi_habit=(-1)*(y_block_condi<=0); %[0,-1]
        % trace of degree of goal-directed (here, only for one subject as an example)
        if(jj2==disp_opt.sbj_ind_for_goal_directed_trace)
            txt_disp=['deg. of goal-directed: ' '''' SBJ{1,jj2}.name ''''];
            figure('Name',txt_disp);   plot(x_mat,y_mat_disp,'k-',x_mat,y_block_condi_goal,'r-.',x_mat,y_block_condi_habit,'b-.',x_mat,zeros(1,length(x_mat)),'-'); axis([min(x_mat) max(x_mat) -2 2])
        end
        % amount of goal-directed in goal-directed blocks and in habitual blocks
        y_goal_amount_in_goalblock=sum(y_mat(find(y_block_condi_goal==1))); % NegLogLik
        y_goal_amount_in_habitualblock=sum(y_mat(find(y_block_condi_habit==-1))); % NegLogLik
        all_y_goal_amount_in_goalblock=[all_y_goal_amount_in_goalblock; y_goal_amount_in_goalblock];
        all_y_goal_amount_in_habitualblock=[all_y_goal_amount_in_habitualblock; y_goal_amount_in_habitualblock];
        
        
        %% 1. correlation btw degree of goal-driected and each regressor
        num_regressors=size(SBJ{1,jj2}.regressor,2);
        list_testing_regressor=[1 2 3 4 7 8 9]; % index in SBJ.regressor : spe,rpe,U1,U2,invF1,invF2,W1
        i_r0=0; vec_all=[];
        vec_a=y_mat;
        for i_r=list_testing_regressor(1:2) % SPE, RPE regressor
            i_r0=i_r0+1;
            vec_b=SBJ{1,jj2}.regressor{1,i_r}.value(row_mat(i_r),:);
            vec_b=(vec_b-min(vec_b))/(max(vec_b)-min(vec_b))-0.5; % [-0.5,0.5]
            list_legend{i_r0,1}=SBJ{1,jj2}.regressor{1,i_r}.name;
            [c_val lag]=xcorr(vec_a,vec_b,'coeff');            out_corr(i_r0)=c_val(find(lag==0)); % zero-lag point = correlation
        end
        % U1 - U2
        i_r0=i_r0+1;    list_legend{i_r0,1}='Uncertainty(M1-M2)';
        vec_b=SBJ{1,jj2}.regressor{1,list_testing_regressor(3)}.value(row_mat(list_testing_regressor(3)),:)-...
            SBJ{1,jj2}.regressor{1,list_testing_regressor(4)}.value(row_mat(list_testing_regressor(4)),:);
        vec_b=vec_b/max(abs(vec_b)); %[-1,1]
        [c_val lag]=xcorr(vec_a,vec_b,'coeff');        out_corr(i_r0)=c_val(find(lag==0));
        % invF1 - invF2
        i_r0=i_r0+1;    list_legend{i_r0,1}='invFano(M1-M2)';
        vec_b=SBJ{1,jj2}.regressor{1,list_testing_regressor(5)}.value(row_mat(list_testing_regressor(5)),:)-...
            SBJ{1,jj2}.regressor{1,list_testing_regressor(6)}.value(row_mat(list_testing_regressor(6)),:);
        vec_b=vec_b/max(abs(vec_b)); %[-1,1]
        [c_val lag]=xcorr(vec_a,vec_b,'coeff');        out_corr(i_r0)=c_val(find(lag==0));
        % W1
        i_r0=i_r0+1;
        i_r=list_testing_regressor(7);
        vec_b=SBJ{1,jj2}.regressor{1,i_r}.value(row_mat(i_r),:); % [0,1] because this is probability
        vec_b=2*vec_b-1; % [-1,1]
        list_legend{i_r0,1}='Prob_{M1}';%SBJ{1,jj2}.regressor{1,i_r}.name;
        [c_val lag]=xcorr(vec_a,vec_b,'coeff');            out_corr(i_r0)=c_val(find(lag==0)); % zero-lag point = correlation
        % collecting values
        all_out_corr=[all_out_corr; out_corr];
        
    end
    
    
    figure('Name','degree of goal-directed in goal-directed/habit blocks');
    % (1) error bar plot
    %     % mean of NegLogLik test
    %     y_plot1=[mean(all_y_goal_amount_in_goalblock) mean(all_y_goal_amount_in_habitualblock)];
    %     % 95% confidence interval
    %      [h,p_fwd,ci_fwd] = ttest(all_y_goal_amount_in_goalblock);  [h,p_sarsa,ci_sarsa] = ttest(all_y_goal_amount_in_habitualblock);
    %     errY1=[sum(abs(ci_fwd)) sum(abs(ci_sarsa))]/2;
    %     barwitherr(errY1, y_plot1);
    %     set(gca,'XTickLabel',{'goal block','habit block'})
    %     ylabel('deg of goal-directed')
    % (2) boxlot
    y_plot1=[(all_y_goal_amount_in_goalblock) (all_y_goal_amount_in_habitualblock)];
    [h_fwd,p_fwd,ci_fwd] = ttest(all_y_goal_amount_in_goalblock);  [h_sarsa,p_sarsa,ci_sarsa] = ttest(all_y_goal_amount_in_habitualblock);
    pass_fwd='';    pass_sarsa='';
    if(h_fwd==1)    pass_fwd='*'; end % add '*' if reject the null hypothesis at the 5% significance level.
    if(h_sarsa==1)    pass_sarsa='*'; end % add '*' if reject the null hypothesis at the 5% significance level.
    disp_txt{1,1}=['goal block' sprintf('\n') '(' pass_fwd sprintf('p=%1.1e)',p_fwd)]; disp_txt{1,2}=['habit block' sprintf('\n') '(' pass_sarsa sprintf('p=%1.1e)',p_sarsa)];
    boxplot(y_plot1,disp_txt)
    ylabel('deg of goal-directed')
    
    figure('Name','Corr. btw (degree of goal-directed) and regressors');
    % (1) error bar plot
    %     y_plot1=mean(all_out_corr);
    %     errY1=[];
    %     for j=1:1:size(all_out_corr,2)
    %         [h,p,ci_corr] = ttest(all_out_corr(:,j)-mean(all_out_corr(:,j))); % 95% confidence interval
    %         errY1=[errY1 sum(abs(ci_corr))/2];
    %     end
    %     barwitherr(errY1, y_plot1);
    %     set(gca,'XTickLabel',list_legend)
    %     ylabel('correlation')
    % (2) boxlot
    y_plot1=[all_out_corr];
    for j=1:1:size(all_out_corr,2)
        [h,p(j),ci_corr] = ttest(all_out_corr(:,j)); % 95% confidence interval
        pass_txt='';
        if(h==1)    pass_txt='*'; end % add '*' if reject the null hypothesis at the 5% significance level.
        list_legend_disp{1,j}=[list_legend{j,1}, sprintf('\n') '(' pass_txt sprintf('p=%1.1e)',p(j))];
    end
    boxplot(y_plot1,list_legend_disp);
    ylabel('correlation')
    
    
    %% Behavioral analysis #2 (intuitive version) - "how often sbj switches his choice in state4 in habitual block?"
    if(1)
        all_beh_perc=cell(1,size(SBJ,2));
        all_beh_perc_determined=cell(1,size(SBJ,2));
        cond_indi_list=[-1, 6, 7, 8];
        for jj2=1:1:size(SBJ,2)        % each subject
            all_beh_perc0=zeros(4,5); % percentage of left choice in 4 conditions(h,g10,g20,g40) x 5states. thus '0' for always-left, '1' for always-right
            all_beh_perc0_cnt=zeros(4,5);
            num_sess=size(SBJ{1,jj2}.HIST_behavior_info,2);
            for kk2=1:1:num_sess % each session
                mat_work0=SBJ{1,jj2}.HIST_behavior_info{1,kk2};
                for ii2=1:1:size(mat_work0,1) % each trial
                    condi0=mat_work0(ii2,end); % condition
                    for aa2=1:1:2 % 1st/2nd decision state
                        i_c=find(cond_indi_list==condi0); % condition
                        i_s=mat_work0(ii2,3+aa2); % state
                        i_a=mat_work0(ii2,6+aa2); % action
                        all_beh_perc0_cnt(i_c,i_s)=all_beh_perc0_cnt(i_c,i_s)+1;
                        if(i_a==2) % +1 if right choice is made
                            all_beh_perc0(i_c,i_s)=all_beh_perc0(i_c,i_s)+1;
                        end
                    end
                end
            end
            all_beh_perc0_cnt(find(all_beh_perc0_cnt==0))=1; % make the denominator 1 for states never visited.
            all_beh_perc0=all_beh_perc0./all_beh_perc0_cnt;% normalized and percentized
            all_beh_perc_determined0=abs(all_beh_perc0-0.5)*2; % 1: keep pressing the same button, 0: switching equally
            
            all_beh_perc{1,jj2}=all_beh_perc0;
            all_beh_perc_determined{1,jj2}=all_beh_perc_determined0;
        end
        % collect habitual block's assessment
        box0=[];
        for jj2=1:1:size(SBJ,2)        % each subject
            %         box0=[box0; all_beh_perc{1,jj2}(1,4)];
            box0=[box0; all_beh_perc_determined{1,jj2}(1,4)];
        end
        figure('Name','percent of the consistent choice in state4 in habitual blocks');
        [h,p(j),ci_corr] = ttest(box0); % 95% confidence interval
        pass_txt='';
        if(h==1)    pass_txt='*'; end % add '*' if reject the null hypothesis at the 5% significance level.
        list_legend0{1,1}=[list_legend{j,1}, sprintf('\n') '(' pass_txt sprintf('p=%1.1e)',p(j))];
        boxplot(box0,list_legend0)
        %         barwitherr(var(box0), mean(box0));    % Plot with errorbars
        %         legend(list_legend0)
        colormap(flipud(gray)) % flipped gray map
        axis([0.5 1.5 0 1.1])
    end
    
    
    
    %% Behavioral analysis #3 (how much arbitration signal reflects subject's actual choice switching?)
    list_r_ind=[9];%[1:1:10]; % index of regressor to look (SPE,RPE,U1,U2,m1,m2,invF1,invF2,w1,w2)
    list_r_test_row=[7 7 8 8 8 8 8 8 7 7]; % corresponding row in the regressor vector
    list_r_test_trial_s={[2 3]',[2 3]',[2 3]',[2 3]',[2 3]',[2 3]',[2 3]',[2 3]',[1 2 3]',[1 2 3]'}; % trial_s set in which regressor signal to be extracted
    list_block_test=[1 2 3 4]; % [1 2] - which block condition included for this analysis
    list_block_test_next=[2 1 4 3]; % [2 1] - the block condition should come right after the 'list_block_test'.
    
    % 1:percentile-based bin, 2:equal-sized bin (NOTE:percentile-based method is not ideal option because the corresponding threshold is driven by distribution bias!)
    option_bin=2;
    list_percentile=[0 50 100]; % for option1
    bin_size=3; % for option2, = # of threshold of bins (=# of bins +1)
    list_bin_val=[0 .5 1]; % for option 3
    
    
    % 1. obtain percentile-threshold from the regressor vector
    all_r_val=cell(size(SBJ,2),length(list_r_ind));
    all_r_threshold_percentile=cell(size(SBJ,2),length(list_r_ind));    all_r_threshold_equal_bin=cell(size(SBJ,2),length(list_r_ind)); all_r_threshold_specified_bin=cell(size(SBJ,2),length(list_r_ind));
    for jj2=1:1:size(SBJ,2)        % each subject
        for rr2=1:1:length(list_r_ind) % each regressor
            %             disp(sprintf('[sbj-%02d/%02d] obtainning the threshold from the regressor vector (%02d/%02d)',jj2, size(SBJ,2),rr2,length(list_r_ind)));
            % (1) threshold for percentile-based bind
            line_ind=SBJ{1,jj2}.regressor{1,list_r_ind(rr2)}.value(4,:);
            if(length(list_r_test_trial_s{1,rr2})==1)
                col_all0=find((line_ind==list_r_test_trial_s{1,rr2}(1))); % extract all corresponding trial_s's column
            end
            if(length(list_r_test_trial_s{1,rr2})==2)
                col_all0=find((line_ind==list_r_test_trial_s{1,rr2}(1))|(line_ind==list_r_test_trial_s{1,rr2}(2))); % extract all corresponding trial_s's column
            end
            if(length(list_r_test_trial_s{1,rr2})==3)
                col_all0=find((line_ind==list_r_test_trial_s{1,rr2}(1))|(line_ind==list_r_test_trial_s{1,rr2}(2))|(line_ind==list_r_test_trial_s{1,rr2}(3))); % extract all corresponding trial_s's column
            end
            all_r_val{jj2,rr2}=SBJ{1,jj2}.regressor{1,list_r_ind(rr2)}.value(list_r_test_row(rr2),col_all0);
            all_r_threshold_percentile{jj2,rr2}=prctile(all_r_val{jj2,rr2},list_percentile);
            % (2) threshold for equal-sized bin
            all_r_threshold_equal_bin{jj2,rr2}=[min(all_r_val{jj2,rr2}):(max(all_r_val{jj2,rr2})-min(all_r_val{jj2,rr2}))/(bin_size-1):max(all_r_val{jj2,rr2})];
            % (3) specified bin
            all_r_threshold_specified_bin{jj2,rr2}=list_bin_val;
        end
    end
    
    %2. measure the deg of choice switching
    if(option_bin==1)        all_r_threshold_use=all_r_threshold_percentile;    end
    if(option_bin==2)        all_r_threshold_use=all_r_threshold_equal_bin;     end
    if(option_bin==3)        all_r_threshold_use=all_r_threshold_specified_bin;     end
    all_beh_perc_Arb=cell(length(list_r_ind),size(SBJ,2));
    all_beh_perc_determined_Arb=cell(length(list_r_ind),size(SBJ,2));
    all_beh_empty_cell_index=cell(length(list_r_ind),size(SBJ,2));
    all_beh_choice_cnt=cell(length(list_r_ind),1);    all_beh_sample_cnt=cell(length(list_r_ind),1);    all_beh_perc_determined_Arb0=cell(length(list_r_ind),1);% just collect for all sbjs
    % define the block set in which you measure choice consistency
    list_ref_blk=[list_block_test;  list_block_test_next];
    for jj2=1:1:size(SBJ,2)        % each subject
        num_sess=size(SBJ{1,jj2}.HIST_behavior_info,2);
        SBJ{1,jj2}.block_set=cell(1,num_sess);
        for kk2=1:1:num_sess % each session
            tmp_mat_test=abs(SBJ{1,jj2}.HIST_block_condition{1,kk2}(2,2:end)-SBJ{1,jj2}.HIST_block_condition{1,kk2}(2,1:end-1));
            [ind_chg0]=find(tmp_mat_test~=0); % means there is a change in col btw [ind_chg(x) ind_chg(x)+1].
            ind_chg=[0 ind_chg0];
            ind_valid=zeros(1,length(ind_chg));
            for gg2=2:1:length(ind_chg)
                condi_chk=SBJ{1,jj2}.HIST_block_condition{1,kk2}(2,[ind_chg(gg2) ind_chg(gg2)+1]);
                oo=find(sum(abs(list_ref_blk-repmat(condi_chk',[1 size(list_ref_blk,2)])))==0);
                if(length(oo)>0) % valid change
                    ind_valid(gg2)=1;
                end
            end
            ms_on=0;
            block_set=[];
            for gg2=[1:1:length(ind_chg)-1] % get start index and end index of block sets to be measured
                if((ind_valid(gg2)==0)&(ind_valid(gg2+1)==1)) % index start
                    ind_st0=ind_chg(gg2)+1;                    ms_on=1;
                end
                if((ind_valid(gg2)==1)&(ind_valid(gg2+1)==0)) % index end
                    ind_end0=ind_chg(gg2+1);                    ms_on=0;
                    block_set=[block_set; [ind_st0 ind_end0]];
                end
            end
            SBJ{1,jj2}.block_set{1,kk2}=block_set; % # of testing blocks x 2. row: testing block index, col(1/2): starting/ending trial index of the testing block
        end
    end
    
    for rr2=1:1:length(list_r_ind) % each regressor
        all_beh_choice_cnt{rr2,1}=zeros(length(all_r_threshold_use{jj2,rr2})-1,5);        all_beh_sample_cnt{rr2,1}=zeros(length(all_r_threshold_use{jj2,rr2})-1,5);
        all_beh_perc_Arb_name{1,rr2}=SBJ{1,1}.regressor{1,list_r_ind(rr2)}.name;
        disp(sprintf('- measure the deg of choice switching based on percentile-threshold of regressor (%02d/%02d)...',rr2,length(list_r_ind)));
        for jj2=1:1:size(SBJ,2)        % each subject
            all_beh_perc0=zeros(length(all_r_threshold_use{jj2,rr2})-1,5); % percentage of left choice in conditions(# of bins of regressor) x 5states. thus '0' for always-left, '1' for always-right
            all_beh_perc0_cnt=zeros(length(all_r_threshold_use{jj2,rr2})-1,5);
            all_beh_empty_ind=ones(length(all_r_threshold_use{jj2,rr2})-1,5); % zero if empty
            num_sess=size(SBJ{1,jj2}.HIST_behavior_info,2);
            for kk2=1:1:num_sess % each session
                mat_work0=SBJ{1,jj2}.HIST_behavior_info{1,kk2};
                for pp2=1:1:size(SBJ{1,jj2}.block_set{1,kk2},1) % each block set
                    for ii2=[SBJ{1,jj2}.block_set{1,kk2}(pp2,1):1:SBJ{1,jj2}.block_set{1,kk2}(pp2,2)]
                        for aa2=1:1:2 % 1st/2nd decision state
                            % identify the condition (=bin# of the regressor signal)
                            tmp_id=[kk2 mat_work0(ii2,1) mat_work0(ii2,2) list_r_test_trial_s{1,rr2}(aa2)]'; % [session block trial state]
                            mycol=find(sum(abs(SBJ{1,jj2}.regressor{1,list_r_ind(rr2)}.value(1:4,:)-repmat(tmp_id,[1 length(SBJ{1,jj2}.regressor{1,list_r_ind(rr2)}.value(1,:))])))==0);
                            val_r0=SBJ{1,jj2}.regressor{1,list_r_ind(rr2)}.value(list_r_test_row(rr2),mycol);
                            mycol1=find((all_r_threshold_use{jj2,rr2}-val_r0)<=0);
                            i_c=mycol1(end); % my condition = my bin#
                            if(i_c==length(all_r_threshold_use{jj2,rr2}))                                i_c=i_c-1;                            end % the biggest value
                            i_s=mat_work0(ii2,3+aa2); % state
                            i_a=mat_work0(ii2,6+aa2); % action
                            all_beh_perc0_cnt(i_c,i_s)=all_beh_perc0_cnt(i_c,i_s)+1;
                            if(i_a==2) % +1 if right choice is made
                                all_beh_perc0(i_c,i_s)=all_beh_perc0(i_c,i_s)+1;
                            end
                        end
                    end
                end
            end
            % find empty slots
            [r99 c99]=find(all_beh_perc0_cnt==0);
            for r22=1:1:length(r99)     all_beh_empty_ind(r99(r22),c99(r22))=0;   end
            all_beh_empty_cell_index{rr2,jj2}=all_beh_empty_ind;
            % save to each cell
            all_beh_perc0_original=all_beh_perc0;
            all_beh_perc0_cnt_original=all_beh_perc0_cnt;
            all_beh_choice_cnt{rr2,1}=all_beh_choice_cnt{rr2,1}+all_beh_perc0_original;
            all_beh_sample_cnt{rr2,1}=all_beh_sample_cnt{rr2,1}+all_beh_perc0_cnt_original;
            
            %             all_beh_perc0_cnt(find(all_beh_perc0_cnt==0))=1; % make the denominator 1 for states never visited.
            %             all_beh_perc0=all_beh_perc0./all_beh_perc0_cnt;% normalized
            %             all_beh_perc_determined0=abs(all_beh_perc0-0.5)*2; % 1: keep pressing the same button, 0: switching equally
            %
            %             all_beh_perc_Arb{rr2,jj2}=all_beh_perc0;
            %             all_beh_perc_determined_Arb{rr2,jj2}=all_beh_perc_determined0;
            
        end
        % check if there is any empty case
        chk_empty=ones(size(all_beh_empty_cell_index{1,1},1),size(all_beh_empty_cell_index{1,1},2));
        for jj2=1:1:size(SBJ,2)
            chk_empty=chk_empty+all_beh_empty_cell_index{rr2,jj2};
        end
        [e1 e2]=find(chk_empty==0);
        if(length(e1)>0)
            error('sample size for each condition is not enough! check!')
        else
            tmp=all_beh_choice_cnt{rr2,1}./all_beh_sample_cnt{rr2,1};
            all_beh_perc_determined_Arb0{rr2,1}=abs(tmp-0.5)*2;
        end
        
    end
    
    
    % display
    ind_r_disp=list_r_ind;%[1 2 3 5 7 9];%[1 2 3 4 5 6 7 8 9 10]; % index of regressor to look (SPE,RPE,U1,U2,m1,m2,invF1,invF2,w1,w2)
    ind_state_disp=[1:1:5]; % state 1~5
    label_bin=cell(1,(bin_size-1));
    bin_ind=round([0:100/(bin_size-1):100]);
    for cc2=1:1:(bin_size-1)             label_bin{1,cc2}=sprintf('level%d',cc2);             end
    %      end
    for rr2=1:1:length(ind_r_disp) % each regressor
        figure('Name',['choice consistency -' all_beh_perc_Arb_name{1,rr2}]);
        for ss2=1:1:length(ind_state_disp) % each state
            label_disp{1,ss2}=sprintf('state%d',ind_state_disp(ss2));
        end
        mat_disp=all_beh_perc_determined_Arb0{rr2,1}; % bin x state
        err_y=zeros(size(mat_disp,1),size(mat_disp,2));
        bar(mat_disp')
        set(gca,'XTickLabel',label_disp)
        legend(label_bin);
        xlabel('state')
        ylabel('choice consistency')
        axis([0.5 5.5 0 1])
        colormap(flipud(summer)) % flipped gray map
        
        figure('Name',['change of the choice consistency (across all states) -' all_beh_perc_Arb_name{1,rr2}]);
        % the baseline MUST be equalized over all states (due to
        % deifferent level of choice consistency in each state)
        mat_disp2=mat_disp;
        for kk2=1:1:size(mat_disp,2)
            mat_disp2(:,kk2)=mat_disp(:,kk2)/mat_disp(1,kk2)*100; % (%)
        end
        y=mean(mat_disp2');
        err_y=zeros(size(y,1),size(y,2));
        for kk2=1:1:size(mat_disp2,1)         err_y(kk2)=std(mat_disp2(kk2,:)');   end
        barwitherr(err_y, y);    % Plot with errorbars
        set(gca,'XTickLabel',label_bin)
        xlabel('signal level')
        ylabel('change of choice consistency (%)')
        axis([0.5 length(y)+.5 0 110])
        colormap(flipud(summer)) % flipped gray map
        title_str=[];
        for aa2=1:1:length(y)
            for bb2=aa2+1:1:length(y)
                [h,p,ci_corr] = ttest(mat_disp2(1,:)-mat_disp2(2,:)); % 95% confidence interval
                pass_txt='';
                if(h==1)    pass_txt='*'; end % add '*' if reject the null hypothesis at the 5% significance level.
                title_str=[title_str [sprintf('[%d-%d] ',aa2,bb2) pass_txt sprintf('p=%1.1e',p) ' || ']];
            end
        end
        title(title_str);
    end
    
end





%% Create regressors
% state. 0.5: fixation mark on, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5,
% 6(+/-)0.1: O1(with win/lost msg), 7(+/-)0.1: O2(with win/lost msg), 8(+/-)0.1: O3(with win/lost msg), 9: O4,
% 10:A1, 11:A2, 20: a short blank page display, -99:fail to choose in time limit, (-) when display off
disp('22');
SBJ_event=SBJ;
SBJ=SBJ2;

if(Do_create_regressors==1)
    
    for jj2=1:1:size(SBJ,2)        % each subject
        disp(sprintf('##### creating regressor structures (sbj%02d/%02d) #######',jj2,size(SBJ,2)));
        SBJ{1,jj2}.HIST_behavior_info_Tag = SBJ_event{1,jj2}.HIST_behavior_info_Tag;
        SBJ{1,jj2}.HIST_event_info = SBJ_event{1,jj2}.HIST_event_info;
        SBJ{1,jj2}.HIST_event_info_Tag = SBJ_event{1,jj2}.HIST_event_info_Tag;
        SBJ{1,jj2}.HIST_block_condition_Tag = SBJ_event{1,jj2}.HIST_block_condition_Tag;
        for kk2=1:1:size(SBJ{1,jj2}.HIST_behavior_info,2)  % each main session
            
            
            
            mat_work=SBJ{1,jj2}.HIST_event_info{1,kk2};
            num_tot_events=size(SBJ{1,jj2}.HIST_behavior_info{1,kk2},1);
            
            
            
            %% 1. Regressor cue presentation - with parametric modulation (timing: stimulus onset)
            % [0-duration] SPE, RPE
            % [RT-duration] Q_fwd, Q_sarsa
            
            % regressor generation for each main session and save it to a single file that is compatible with SPM
            ind_reg=0;  % corresponds to the size of the structure
            ind_reg_abs=0; % actual number of regressors (including parametric)
            durations={};
            onsets={};
            names={};
            pmod=struct('name',{},'param',{},'poly',{});
            
            
            use_model_regressor_cue=1;
            
            % (1) durations, name, onset
            [tmp col_on]=find((mat_work(7,:)==1)|(mat_work(7,:)==2)|(mat_work(7,:)==3)|(mat_work(7,:)==4)|(mat_work(7,:)==5)...
                |(mat_work(7,:)==5.9)|(mat_work(7,:)==6.1)|(mat_work(7,:)==6.9)|(mat_work(7,:)==7.1)|...
                (mat_work(7,:)==7.9)|(mat_work(7,:)==8.1)|(mat_work(7,:)==9));
            if(length(col_on)~=num_tot_events*3)
                error('-ERROR: variable ''mat_work'' missed some event extraction. check!')
            end
            
            RT_mat=[];  onset_mat=[];
            prev_trial=0;  show_n_th_times_t=0;
            param_mat{1,1}=zeros(length(ind_regressor_type_base{1,1}.ind_reg),length(col_on));
            param_mat{1,2}=zeros(length(ind_regressor_type_base{1,2}.ind_reg),length(col_on));
            
            for ll2=1:1:length(col_on)
                
                if(ll2<length(col_on)) % usual case
                    pt_on=mat_work(4,col_on(ll2));
                    col_off=col_on(ll2)+1;
                    %                 col_off=col_on(ll2)-1+find(mat_work(7,[col_on(ll2):1:(col_on(ll2)+2)])==0.5); % find the next fixation mark presentation
                    pt_off=mat_work(4,col_off);
                    RT=pt_off-pt_on;
                else % last event in the session is the outcome presentation
                    RT=2.0;
                end
                RT_mat=[RT_mat RT];
                onset_t=mat_work(4,col_on(ll2));
                onset_mat=[onset_mat onset_t];
                
                % fill out regresssor values
                if(use_model_regressor_cue==1)
                    
                    % regressor type1
                    for nn=1:1:length(ind_regressor_type_base{1,1}.ind_reg)
                        mysession=kk2;
                        myblock=mat_work(1,col_on(ll2)); % block in session
                        mytrial=mat_work(2,col_on(ll2)); % trial in block
                        mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial (arbitration index: 1 at the second stage, 2 at the third stage)
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        param_mat{1,1}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,1}.ind_reg(nn)),col_event);
                        
                    end
                    
                    
                    % regressor type2
                    for nn=1:1:length(ind_regressor_type_base{1,2}.ind_reg)
                        mysession=kk2;
                        myblock=mat_work(1,col_on(ll2)); % block in session
                        mytrial=mat_work(2,col_on(ll2)); % trial in block
                        mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        param_mat{1,2}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,2}.ind_reg(nn)),col_event);
                    end
                    
                end
                
            end
            
            
            % 1st regressors
            ind_reg=ind_reg+1;
            onsets{1,ind_reg}=onset_mat;
            if(reg_type_go_first(1)==1)            names{1,ind_reg}=['Cue_{0T}'];  durations{1,ind_reg}=0;        end
            if(reg_type_go_first(1)==1.5)        names{1,ind_reg}=['Cue_{1T}'];    durations{1,ind_reg}=RT_mat;       end
            
            ind_reg_abs=1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
            
            
            % (2) pmod: how many times each cue presented
            if(use_model_regressor_cue==1)
                for nn=1:1:length(ind_regressor_type_base{1,1}.ind_reg)
                    pmod(1,ind_reg).name{1,nn}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.name];
                    pmod(1,ind_reg).poly{1,nn}=1;
                    pmod(1,ind_reg).param{1,nn}=param_mat{1,1}(nn,:);
                    ind_reg_abs=ind_regressor_type_base{1,1}.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,nn};
                end
            end
            
            
            % 2nd regressors
            ind_reg=ind_reg+1;
            onsets{1,ind_reg}=onset_mat;
            if(reg_type_go_first(2)==1)            names{1,ind_reg}=['Cue_{0T}'];  durations{1,ind_reg}=0;        end
            if(reg_type_go_first(2)==1.5)        names{1,ind_reg}=['Cue_{1T}'];    durations{1,ind_reg}=RT_mat;       end
            ind_reg_abs=ind_regressor_type_base{1,1}.abs_pos_in_design_mat(end)+1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
            
            % (2) pmod:
            if(use_model_regressor_cue==1)
                ind_reg_param=0;
                for nn=1:1:length(ind_regressor_type_base{1,2}.ind_reg)
                    ind_reg_param=ind_reg_param+1;
                    pmod(1,ind_reg).name{1,ind_reg_param}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(ind_reg_param)}.name];
                    pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                    pmod(1,ind_reg).param{1,ind_reg_param}=param_mat{1,2}(ind_reg_param,:);
                    ind_reg_abs=ind_regressor_type_base{1,2}.abs_pos_in_design_mat(ind_reg_param);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                end
                % Reserve spaces for extra regressors (will come from the dummy design matrix)
                if(reg_type_go_first(3)==2)
                    for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=ind_regressor_type_dummy.name{1,nn};
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=rand(1,length(param_mat{1,1}(1,:)));
                        %                         ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                end
            end
            
            
            onset_event=onsets;
            
            
            %% 1-A. Regressor cue presentation - regressor type 3 only!!! with parametric modulation (timing: decision time)
            % [0-duration]
            % [RT-duration]
            
            if(reg_type_go_first(3)==1.7)
                
                use_model_regressor_cue=1;
                
                % (1) durations, name, onset
                [tmp col_on]=find((mat_work(7,:)==10)|(mat_work(7,:)==11));
                
                RT_mat=[];  onset_mat=[];
                prev_trial=0;  show_n_th_times_t=0;
                param_mat3{1,1}=zeros(length(ind_regressor_type_base{1,3}.ind_reg),length(col_on));
                
                
                for ll2=1:1:length(col_on)
                    
                    if(ll2<length(col_on)) % usual case
                        pt_on=mat_work(4,col_on(ll2));
                        col_off=col_on(ll2)+1;
                        %                 col_off=col_on(ll2)-1+find(mat_work(7,[col_on(ll2):1:(col_on(ll2)+2)])==0.5); % find the next fixation mark presentation
                        pt_off=mat_work(4,col_off);
                        RT=pt_off-pt_on;
                    else % last event in the session is the outcome presentation
                        RT=2.0;
                    end
                    RT_mat=[RT_mat RT];
                    onset_t=mat_work(4,col_on(ll2));
                    onset_mat=[onset_mat onset_t];
                    
                    % fill out regresssor values
                    if(use_model_regressor_cue==1)
                        
                        % regressor type3 only
                        for nn=1:1:length(ind_regressor_type_base{1,3}.ind_reg)
                            mysession=kk2;
                            myblock=mat_work(1,col_on(ll2)); % block in session
                            mytrial=mat_work(2,col_on(ll2)); % trial in block
                            mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial (arbitration index: 1 at the second stage, 2 at the third stage)
                            mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(nn)}.value(1:4,:);
                            identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                            col_event=find(identity_tmp==0);
                            param_mat3{1,1}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,3}.ind_reg(nn)),col_event);
                        end
                        
                    end
                    
                end
                
                
                % 3rd regressors
                ind_reg=ind_reg+1;
                onsets{1,ind_reg}=onset_mat;
                if(reg_type_go_first(3)==1.7)            names{1,ind_reg}=['Decision_{0T}'];  durations{1,ind_reg}=0;        end
                
                ind_reg_abs=ind_regressor_type_base{1,2}.abs_pos_in_design_mat(end)+1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
                
                
                % (2) pmod:
                if(use_model_regressor_cue==1)
                    ind_reg_param=0;
                    for nn=1:1:length(ind_regressor_type_base{1,3}.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(ind_reg_param)}.name];
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=param_mat3{1,1}(ind_reg_param,:);
                        ind_reg_abs=ind_regressor_type_base{1,3}.abs_pos_in_design_mat(ind_reg_param);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                    % Reserve spaces for extra regressors (will come from the dummy design matrix)
                    for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=ind_regressor_type_dummy.name{1,nn};
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=rand(1,length(param_mat3{1,1}(1,:)));
                        %                         ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                end
                
            end
            
            %%
            
            
            
            
            % (3) Saving normal regressor file
            tot_num_myregressor=length(list_name_for_contrast);
            save_file_name=['Regressor--' SBJ{1,jj2}.name '_sess' sprintf('%02d.mat',kk2)];
            if(Is_save_files_local==1)
                save([save_path_result '/' save_file_name],'durations', 'names', 'onsets', 'pmod' );
                %                 eval(['save ' save_for_SPM save_file_name ' durations names onsets pmod'])
            end
            if(Is_save_files_cluster==1)
                %                 eval(['save ' save_path_neuroecon save_file_name ' durations names onsets pmod'])
            end
            
            
            
            %% 2. Extra Regressors independent of cue presentation (timing: TR)
            % The regressors will be saved in separate .mat file.
            
            % regressor generation for each main session and save it to a single file that is compatible with SPM
            ind_reg=0;  % corresponds to the size of the structure
            ind_reg_abs=0; % actual number of regressors (including parametric)
            durations={};
            onsets={};
            names={};
            pmod=struct('name',{},'param',{},'poly',{});
            
            
            ind_reg=ind_reg+1;
            TR_CBIC=2.78; % (sec)
            % (1) duration,onset,name
            durations{1,ind_reg}=TR_CBIC;
            onsets{1,ind_reg}=[0:TR_CBIC:(mat_work(4,end)+20)];
            names{1,ind_reg}=['Dummy'];
            % the name will NOT be added to "list_name_for_contrast" because it is a dummy regressor.
            length_reg=length(onsets{1,1});
            % (2-1) determine the regressor values
            regressor_extra=zeros(length(ind_regressor_type_dummy.ind_reg),length_reg);
            for nn=1:1:length(ind_regressor_type_dummy.ind_reg) % for each regressor
                % find the previous onset point closest to the current time
                t_scan=0;
                for i_scan=1:1:length_reg
                    % find the recent onset (event) time on which the regressor value has been updated.
                    test_mat=onset_event{1,1}-t_scan;
                    test_ii=find(test_mat<0);
                    if(length(test_ii)==0) % first a few scans before the event
                        myscan_time=onset_event{1,1}(1);
                    else % use the value of previous event
                        myscan_time=onset_event{1,1}(min(length(onset_event{1,1}),test_ii(end)));
                    end
                    % find the corresponding {block#, trial#, and trial_s#} in event matrix
                    mycol=find(abs(mat_work(4,:)-myscan_time)<0.001);
                    mysession=kk2; % session
                    myblock=mat_work(1,mycol); % block in session
                    mytrial=mat_work(2,mycol); % trial in block
                    mytrial_s=mat_work(3,mycol); % trial_s in trial (mytrial_s=max(1,mat_work(3,mycol)-1);)
                    % find the corresponding regressor value in a regressor matrix
                    if(mytrial_s~=1)
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        regressor_extra(nn,i_scan)=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(row_mat(ind_regressor_type_dummy.ind_reg(nn)),col_event);
                    else % mytrial_s=1: the first state
                        if(i_scan==1) % for the very first scan, we simply read the first regressor value.
                            regressor_extra(nn,i_scan)=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(row_mat(ind_regressor_type_dummy.ind_reg(nn)),1);
                        else % for the first state in every trial, we simply take the regressor value at t-1 (because there is no update)
                            regressor_extra(nn,i_scan)=regressor_extra(nn,i_scan-1);
                        end
                    end
                    % compute the next scan time
                    t_scan=t_scan+TR_CBIC;
                end
            end
            % (2-2) pmod: parametric modulators
            for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                pmod(1,ind_reg).name{1,nn}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.name];
                pmod(1,ind_reg).poly{1,nn}=1;
                pmod(1,ind_reg).param{1,nn}=regressor_extra(nn,:);
                ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,nn};
            end
            
            % (3) Saving dummy regressor file
            tot_num_myregressor=length(list_name_for_contrast);
            save_file_name=['Regressor_dummy--' SBJ{1,jj2}.name '_sess' sprintf('%02d.mat',kk2)];
            if(Is_save_files_local==1)
                save([save_path_result '/' save_file_name],'durations','names' ,'onsets', 'pmod', 'ind_regressor_type_dummy');
                
%                 eval(['save ' save_path_result save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
%                 eval(['save ' save_for_SPM save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
            end
            if(Is_save_files_cluster==1)
%                 eval(['save ' save_path_neuroecon save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
            end
            
        end
    end
    
    
    %% Saving Contrast file
    % [index of my regressors for contrast vector] : total main regressor=6, total regressors=7
    clear contrast_spm
    
    total_number_regressor=tot_num_myregressor+6; % # + 6 movements
    ind_contrast_vec=0;
    
    
    % individual : (ex) [0 1 0 0 0 0 0 0]
    for ii=1:1:tot_num_myregressor
        ind_contrast_vec=ind_contrast_vec+1;
        contrast=zeros(1,tot_num_myregressor);
        contrast(1,ii)=1;
        contrast_spm{1,ind_contrast_vec}.name=list_name_for_contrast{1,ii};
        contrast_spm{1,ind_contrast_vec}.vec=contrast;
    end
    
    % % difference : (ex) [0 0 0 0 1 -1 0 0]
    % combination_mat=combnk(param_regressor_type_cue_abs_pos_in_design_mat,2);
    % for kk=1:1:size(combination_mat,1)
    %     % A-B
    %     ind_contrast_vec=ind_contrast_vec+1;
    %     contrast=zeros(1,total_number_regressor);
    %     contrast(1,combination_mat(kk,1))=1;    contrast(1,combination_mat(kk,2))=-1;
    %     contrast_spm{1,ind_contrast_vec}.name=[list_name_for_contrast{1,combination_mat(kk,1)} '>' list_name_for_contrast{1,combination_mat(kk,2)}];
    %     contrast_spm{1,ind_contrast_vec}.vec=contrast;
    %     % B-A
    %     ind_contrast_vec=ind_contrast_vec+1;
    %     contrast=zeros(1,total_number_regressor);
    %     contrast(1,combination_mat(kk,2))=1;    contrast(1,combination_mat(kk,1))=-1;
    %     contrast_spm{1,ind_contrast_vec}.name=[list_name_for_contrast{1,combination_mat(kk,1)} '<' list_name_for_contrast{1,combination_mat(kk,2)}];
    %     contrast_spm{1,ind_contrast_vec}.vec=contrast;
    % end
    
    if(Is_save_files_local==1)
        save([save_path_result '/contrast_spm.mat'],'contrast_spm', 'ind_regressor_type_base', 'ind_regressor_type_dummy', 'param_regressor_type_cue_abs_pos_in_design_mat', 'list_name_for_contrast' );
        
%         eval(['save ' save_path_result 'contrast_spm.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
%         eval(['save ' save_for_SPM 'contrast_spm.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
    end
    if(Is_save_files_cluster==1)
%         eval(['save ' save_path_neuroecon 'contrast_spm.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
    end
    
end








%% measure the degree of habit in habitual conditions
% block condition - 1: G(with low uncertainty), 2: G''(with high uncertainty), 3:H(with high uncertainty), 4:H''(with low uncertainty)';
% ### [note]: using "state_action_vec_ref" might be stupid idea. using the
% actual action taken by the model would make more sense!!!
if(1)
    
    state_action_vec_ref=[1 2; 2 1; 3 2; 4 2; 5 1]; % col1:state, col2:corresponding action
    for i=1:1:num_sbj_included
        num_tot_sess=size(SBJ{1,i}.HIST_behavior_info,2);
        sum_mat_percentage=zeros(4,2);
        for i_sess=1:1:num_tot_sess
            condi_to_check=[1 2 3 4]; %for all conditions
            mat_percentage=[];
            for kk=1:1:length(condi_to_check)
                % (1) haibual condition
                row_condi=find(SBJ{1,i}.HIST_behavior_info{1,i_sess}(:,3)==condi_to_check(kk));
                num_total_trial=size(SBJ{1,i}.HIST_behavior_info{1,i_sess},1);
                val_score=0;
                for j=1:1:length(row_condi)
                    state_vec=SBJ{1,i}.HIST_behavior_info{1,i_sess}(row_condi(j),[4:5]);
                    state_action_vec0=state_action_vec_ref(state_vec,2)'; % strong habitual action
                    state_action_vec1=SBJ{1,i}.HIST_behavior_info{1,i_sess}(row_condi(j),[7:8]); % subject's action
                    eval_vec=abs(state_action_vec0-state_action_vec1); % all zero = same actions
                    val_score=val_score+length(find(eval_vec==0));
                end
                mat_percentage=[mat_percentage; [condi_to_check(kk) 100*val_score/(2*length(row_condi))]];
            end
            SBJ{1,i}.HIST_block_condition_habit_score{1,i_sess}=mat_percentage;
            SBJ{1,i}.HIST_block_condition_habit_score_Tag{1,i_sess}='col1: block condition, col2: percentage of habitual action';
            sum_mat_percentage=sum_mat_percentage+mat_percentage;
        end
        SBJ{1,i}.HIST_block_condition_habit_score_mean=sum_mat_percentage/num_tot_sess;
        
    end
    
end






disp('- all done.')
end