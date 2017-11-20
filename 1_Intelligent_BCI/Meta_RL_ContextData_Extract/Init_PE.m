function [MODEL]=Init_PE(DATA_IN,param_in,mode)
%% pseudoArbitration model : just follows the subjects behavior
% PE 뽑아오기. model rl 코드 기준으로 해야할듯.
SPEhist=[];
RPEhist=[];
map_opt.transition_prob_seed=[0.7 0.3];
map_opt.reward_seed=[40 20 10 0];
[myMap_new N_state N_action N_transition]=Model_Map_Init2('sangwan2012c',map_opt);
myState=Model_RL_Init(N_state,N_action,N_transition);
% myArbitrator = '?';
pop_id=1;
ll=1;

% SARSA
param_sarsa.gamma=1.0; % fixed - not actual parameter
param_sarsa.alpha = param_in(pop_id, 4);
% FWD
param_fwd.alpha = param_in(pop_id, 4);
% Softmax parameter for all models
param_sarsa.tau=param_in(pop_id, 3);
param_fwd.tau=param_in(pop_id, 3);

        
% (1) phase 1 - random action, no reward
state_fwd=myState;  state_fwd.name='fwd';
state_sarsa=myState;    state_sarsa.name='sarsa';
map=myMap_new;  map0=myMap_new; map0_s=myMap_new;

%% pre-training part
kk=1; % originally it is index of simulations
num_max_trial0=size(DATA_IN{1,ll}.HIST_behavior_info_pre{1,1},1);  % size는 trial의 값을 저장하는 것으로.
map0.epoch=kk;                map0_s.epoch=kk;
map0.data=DATA_IN{1,ll}.HIST_behavior_info_pre{1,1};  map0_s.data=DATA_IN{1,ll}.HIST_behavior_info_pre{1,1};
opt_state_space.use_data=1;  % 1; Using subject's behavioral data % use subject data for state-transition

    % fwd sarsa
    i=0; 
    while (i<num_max_trial0)
        i=i+1;
        map0.trial=i;
        Tprob_Ulow=[0.9 0.1];   Tprob_Uhigh=[0.5 0.5];
        Tprob_Umid=[0.7 0.3];

        block_condition=DATA_IN{1,ll}.HIST_block_condition_pre{1,1}(2,i);
        rwd_condition=DATA_IN{1,ll}.HIST_behavior_info_pre{1,1}(i,18);  %여기서는 그냥 goal condition index를 넘겨주고 실제로 model_map_update에서 업데이트 해주는 형태로 해두었음.


        switch block_condition
            case 1
                Tprob0=Tprob_Umid;
            case 2
                Tprob0=Tprob_Ulow;
            case 3
                Tprob0=Tprob_Uhigh;
            case 4
                Tprob0=Tprob_Umid;
        end
        % update T_prob
        map0=Model_Map_update(map0,'T',Tprob0); % map0.action update
        % set Reward
        map0=Model_Map_update(map0,'R',rwd_condition); % map0.reward update


        % initializing the state
        param_fwd0=param_fwd;   param_fwd0.alpha=0.15;
        [state_fwd map0]=StateClear(state_fwd,map0);
        while (~state_fwd.JobComplete)
            % decision
            if(mode.experience_sbj_events(1)==1)
                state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'decision_behavior_data_save');
            else
                state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'decision_random');
            end
            % state transition
            [state_fwd map0]=StateSpace_v1(state_fwd,map0,opt_state_space);  % map&state index ++
            % 1. fwd model update
            state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'fwd_update');
        end
    end

    % sarsa update
    i=0;
    while (i<num_max_trial0)
        i=i+1;
        map0_s.trial=i;

        Tprob_Ulow=[0.9 0.1];   Tprob_Uhigh=[0.5 0.5];
        Tprob_Umid=[0.7 0.3];
        block_condition=DATA_IN{1,ll}.HIST_block_condition_pre{1,1}(2,i);
        rwd_condition=DATA_IN{1,ll}.HIST_behavior_info_pre{1,1}(i,18);  %여기서는 그냥 goal condition index를 넘겨주고 실제로 model_map_update에서 업데이트 해주는 형태로 해두었음.

        switch block_condition
            case 1
                Tprob0=Tprob_Umid;
            case 2
                Tprob0=Tprob_Ulow;
            case 3
                Tprob0=Tprob_Uhigh;
            case 4
                Tprob0=Tprob_Umid;
        end
        % update T_prob
        map0=Model_Map_update(map0,'T',Tprob0);
        % set Reward
        map0=Model_Map_update(map0,'R',rwd_condition);


        % initializing the state
        param_sarsa0=param_sarsa;   param_sarsa0.alpha=param_fwd0.alpha*1.2; % make a bit faster than MB due to the slower convergence rate of MF.
        [state_sarsa map0_s]=StateClear(state_sarsa,map0_s);
        while (~state_sarsa.JobComplete)
            % 0. current action selection : (s,a) - using arbitrator's Q-value
            if(mode.experience_sbj_events(1)==1)
                state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_behavior_data_save');
            else
                state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_random');
            end
            % 1. sarsa state update (get reward and next state) : (r,s')
            [state_sarsa map0_s]=StateSpace_v1(state_sarsa,map0_s,opt_state_space); % map&state index ++
            % 1. sarsa next action selection : (s',a') - if s' is terminal, then no decision
            state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_hypo'); % 이거 때문에 action valid 넣어 줬던 거 같은 데 나중에
            % 1. sarsa model upate
            state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'sarsa_update');
        end
    end
    
%% Main Learning
num_max_session=size(DATA_IN{1,ll}.HIST_behavior_info,2);

SARSA=cell(num_max_session,1);
TTank=cell(num_max_session,1);
% myArbitrator_top=myArbitrator;
mode_data_prev=6;
K=0;
for ind_sess=1:1:num_max_session
    i=0;
    num_max_trial=size(DATA_IN{1,ll}.HIST_behavior_info{1,ind_sess},1);
    map.epoch=kk;
    map.data=DATA_IN{1,ll}.HIST_behavior_info{1,ind_sess};
    map.condition_id=DATA_IN{1,ll}.HIST_block_condition{1,ind_sess}(2,:);
    while (i<num_max_trial)
        i=i+1;
        map.trial=i;
        
        % initializing the state
        [state_fwd map]=StateClear(state_fwd,map);
        [state_sarsa map]=StateClear(state_sarsa,map);
        % read a reward value set from data
        mode_data=map.data(map.trial,18);
        block_condition=DATA_IN{1,ll}.HIST_block_condition{1,ind_sess}(2,map.trial);
        
        switch block_condition
            case 1
            case 2
            case 3
            case 4
        end
        
        if((mode_data_prev ~= mode_data)) % if there is any visual cue change, then goal mode
            myArbitrator_top.backward_flag=1; % after backward update, should set it to 0.
            % 2016 11 07
            % goal mode 에서 habitual mode로 가는 경우는?
        end
       
       % myArbitrator_top.m2_prob=1-myArbitrator_top.m1_prob;
        
        while (~map.JobComplete)
            % index synchronization
            state_fwd.index=map.index;                state_sarsa.index=map.index;
            
            % fwd mode: backward update of fwd model
            if(myArbitrator_top.backward_flag==1) % if the agent detects context change
                % (1) revaluation : reevaluation for goal mode
                map=Model_Map_update(map,'R',mode_data');
                % (2) backward update of the fwd model
                if(mode.USE_BWDupdate_of_FWDmodel==1)
                    state_fwd=Model_RL2(state_fwd, param_fwd, map, 'bwd_update');
                end
            else
                %
            end
            
            
            % compute real Q-value by merging two Q (fwd & sarsa)
            
%             myArbitrator_top.Q=...
%                 ((myArbitrator_top.m1_prob*state_fwd.Q).^myArbitrator_top.p+...
%                 (myArbitrator_top.m2_prob*state_sarsa.Q).^myArbitrator_top.p).^(1/myArbitrator_top.p);
%             
%             myArbitrator_top.Q_old=myArbitrator_top.Q;
            
            %% main computation
            % 0. current action selection : (s,a) - using arbitrator's Q-value
            state_fwd=Model_RL2(state_fwd, param_fwd, map, 'decision_behavior_data_save');
            
            
            % 1. fwd state update (get reward and next state) : (r,s')
            [state_fwd map]=StateSpace_v1(state_fwd,map,opt_state_space); % map&state index ++
            state_sarsa.index=state_fwd.index;
            % 1. fwd model update
            state_fwd=Model_RL2(state_fwd, param_fwd, map, 'fwd_update');
            % 2. state synchronization
            state_sarsa.state_history(state_fwd.index)=state_fwd.state_history(state_fwd.index);
            state_sarsa.SARSA=state_fwd.SARSA;
            % 3. sarsa next action selection : (s',a') - if s' is terminal, then no decision
            state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_hypo');
            % 3. sarsa model upate
            state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'sarsa_update');
            % history synchronization
            myArbitrator_top.state_history=state_fwd.state_history;
            myArbitrator_top.reward_history=state_fwd.reward_history;
            
            temp=[];
            temp2=[];
            temp3=[];
            switch map.index
                case 1
                    temp = [map.index ; state_fwd.SPE_history(map.index)];
                    SPEhist=[SPEhist temp];
                    temp2 = [map.index ; state_sarsa.RPE_history(map.index)]; 
                    RPEhist=[RPEhist temp2];
                    SARSA{ind_sess} = [SARSA{ind_sess} ; state_sarsa.SARSA];
                    

                case 2
                    temp = [map.index ; state_fwd.SPE_history(map.index)];
                    SPEhist=[SPEhist temp];
                    temp2 = [map.index ; state_sarsa.RPE_history(map.index)]; 
                    RPEhist=[RPEhist temp2];
                    SARSA{ind_sess} = [SARSA{ind_sess} ; state_sarsa.SARSA];
                    
                case 3
                    temp = [map.index ; state_fwd.SPE_history(map.index)];
                    SPEhist=[SPEhist temp];
                    temp2 = [map.index ; state_sarsa.RPE_history(map.index)]; 
                    RPEhist=[RPEhist temp2];
                    SARSA{ind_sess} = [SARSA{ind_sess} ; state_sarsa.SARSA];
                    
            end
            
            TTank{ind_sess} = cat(4, TTank{ind_sess} ,state_fwd.T);
            K=K+1;
                        
        end
        mode_data_prev=mode_data;

    end
end

RPEac_set{1} = SARSA;
Tset{1} = TTank;
% 
% 
% % 0. current action selection : (s,a) - using arbitrator's Q-value
% state_fwd=Model_RL2(state_fwd, param_fwd, map, 'decision_behavior_data_save');
% 
% 
% % 1. fwd state update (get reward and next state) : (r,s')
% [state_fwd map]=StateSpace_v1(state_fwd,map,opt_state_space); % map&state index ++
% state_sarsa.index=state_fwd.index;
% % 1. fwd model update
% state_fwd=Model_RL2(state_fwd, param_fwd, map, 'fwd_update');
% % 2. state synchronization
% state_sarsa.state_history(state_fwd.index)=state_fwd.state_history(state_fwd.index);
% state_sarsa.SARSA=state_fwd.SARSA;
% % 3. sarsa next action selection : (s',a') - if s' is terminal, then no decision
% state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_hypo');
% % 3. sarsa model upate
% state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'sarsa_update');
% % history synchronization
% myArbitrator_top.state_history=state_fwd.state_history;
% myArbitrator_top.reward_history=state_fwd.reward_history;



MODEL = struct('SPE', SPEhist, 'RPE', RPEhist, 'RPE_SARSA',RPEac_set, 'SPE_T', Tset, 'trials', K);

end