function [myState]=Model_RL2(myState, myParam, myMap, mode, varargin)

%% extract current/previous state/action
state_current=myState.state_history(myState.index);
action_current=myState.action_history(myState.index);


%% process
switch lower(mode)
    
    %% [random decision(action selection) mode] - (s,a)
    case {'decision_random'}
        check_cond=(myState.action_history(myState.index)==0); % if no decision made previously --
        check_cond0=(myMap.IsTerminal(state_current)~=1);
        if(check_cond)
            if(check_cond0)
                
                myState.action_prob(myState.index)=rand;
                nn_action=size(myState.Q,2);
                myState.action_history(myState.index)=ceil(nn_action*rand);
                
                
                
                
                
                
                                
                % filling in SARSA matrix - filling in (s,a)
                myState.SARSA(1:2)=[myState.state_history(myState.index) myState.action_history(myState.index)];
                myState.SARSA(6)=0; % this means SARSA matrix has not been completed yet.
            end
        else
            disp('#### WARNING: decision made previously. no need to make decision twice. ####');
        end
        
    %% [behavior-directed decision(action selection) mode] - (s,a)
    case {'decision_behavior_data'} % read action from the behavior data record
        check_cond=(myState.action_history(myState.index)==0); % if no decision made previously --
        check_cond0=(myMap.IsTerminal(state_current)~=1);        
        if(check_cond)
            if(check_cond0)
                myState.action_prob(myState.index)=0;                
                myState.action_history(myState.index)=myMap.data{1,myMap.epoch}(myMap.trial,6+myState.index);
                % filling in SARSA matrix - filling in (s,a)
                myState.SARSA(1:2)=[myState.state_history(myState.index) myState.action_history(myState.index)];
                myState.SARSA(6)=0; % this means SARSA matrix has not been completed yet.
            end
        else
            disp('#### WARNING: decision made previously. no need to make decision twice. ####');
        end
        
    case {'decision_behavior_data_save'} % read action from the behavior data record
        % ##(under contruction)## ".index" should be checked!!!! 13 JUNE 2012 #####        
        check_cond=(myState.action_history(myState.index)==0); % if no decision made previously --
        check_cond0=(myMap.IsTerminal(state_current)~=1);
        if(check_cond)
            if(check_cond0)
                myState.action_prob(myState.index)=0;
                myState.action_history(myState.index)=myMap.data(myMap.trial,6+myState.index);
                % filling in SARSA matrix - filling in (s,a)
                myState.SARSA(1:2)=[myState.state_history(myState.index) myState.action_history(myState.index)];
                myState.SARSA(6)=0; % this means SARSA matrix has not been completed yet.
            end
        else
            disp('#### WARNING: decision made previously. no need to make decision twice. ####');
        end
        
     %% [arbitrator-decision(action selection) mode] - (s,a)
    case {'decision_arbitrator'}
        if(size(varargin,2)>=2) % use the arbitrator's Q for decision
            myArbitrator=varargin{1};
            myState2=varargin{2};
            check_cond=(myState.action_history(myState.index)==0); % if no decision made previously --
            check_cond0=(myMap.IsTerminal(state_current)~=1);            
            if(check_cond)
                if(check_cond0)
                    % merge Q value of the fwd and sarsa
                    myArbitrator.m2_prob=1-myArbitrator.m1_prob;
                    if(strcmp(myState2.name,'sarsa')==1) % myState=fwd, myState2=sarsa
                        myArbitrator.Q=...
                            ((myArbitrator.m1_prob*myState.Q).^myArbitrator.p+...
                            (myArbitrator.m2_prob*myState2.Q).^myArbitrator.p).^(1/myArbitrator.p);
                    else % myState=sarsa, myState2=fwd
                        myArbitrator.Q=...
                            ((myArbitrator.m1_prob*myState2.Q).^myArbitrator.p+...
                            (myArbitrator.m2_prob*myState.Q).^myArbitrator.p).^(1/myArbitrator.p);
                    end
                    % softmax                    
                    q_in=(myMap.action_valid(myState.index,:)).*myArbitrator.Q(state_current,:); % only for available actions
                    var_exp=exp(myParam.tau*q_in); % (N_actionx1)
                    Prob_action=var_exp/sum(var_exp); % (N_actionx1)
                    myState.action_prob(myState.index)=Prob_action(1);
                    [bincounts] = histc(rand,[0 cumsum(Prob_action)]);
                    myState.action_history(myState.index)=find(bincounts(1:end-1)>0);                    
                    % filling in SARSA matrix - filling in (s,a)
                    myState.SARSA(1:2)=[myState.state_history(myState.index) myState.action_history(myState.index)];
                    myState.SARSA(6)=0; % this means SARSA matrix has not been completed yet.
                end
            else
                disp('#### WARNING: decision made previously. no need to make decision twice. ####');
            end
        else
            error('- insufficient number of input arguments !')
        end
    
    %% [actual decision(action selection) mode] - (s,a)
    case {'decision'}
        check_cond=(myState.action_history(myState.index)==0); % if no decision made previously --
        check_cond0=(myMap.IsTerminal(state_current)~=1);
        if(check_cond)
            if(check_cond0) % if not in a terminal state                
                q_in=(myMap.action_valid(myState.index,:)).*myState.Q(state_current,:);
                var_exp=exp(myParam.tau*q_in); % (N_actionx1)
                Prob_action=var_exp/sum(var_exp); % (N_actionx1)
                myState.action_prob(myState.index)=Prob_action(1);
                [bincounts] = histc(rand,[0 cumsum(Prob_action)]);
                myState.action_history(myState.index)=find(bincounts(1:end-1)>0);                
                % filling in SARSA matrix - filling in (s,a)
                myState.SARSA(1:2)=[myState.state_history(myState.index) myState.action_history(myState.index)];
                myState.SARSA(6)=0; % this means SARSA matrix has not been completed yet.
            end
        else
            disp('#### WARNING: decision made previously. no need to make decision twice. ####');
        end
        
        
    %% [hypothetical decision(action selection) mode] - (s',a')
    case {'decision_hypo'}
        check_cond0=(myMap.IsTerminal(state_current)~=1);
        if(check_cond0) % if s' is not terminal, myMap.index = 2;           
            q_in=myState.Q(state_current,:);
            var_exp=exp(myParam.tau*q_in); % (N_actionx1)
            Prob_action=var_exp/sum(var_exp); % (N_actionx1)
            [bincounts] = histc(rand,[0 cumsum(Prob_action)]);
            action_hypo=find(bincounts(1:end-1)>0);            
            % filling in SARSA matrix - filling in (s',a')            last column means SARSA matrix has been completed.
            myState.SARSA(5:6)=[action_hypo 1];
        else % if s' is terminal, then no decision has to be made, i.e., myState.SARSA(5)=0. myMap.index = 3;
            % filling in SARSA matrix - filling in (s',a')            last column means SARSA matrix has been completed.
            myState.SARSA(5:6)=[0 1];
        end
        
    case {'decision_dpnmm_sarsa'}
        check_cond0=(myMap.IsTerminal(state_current)~=1);
       if(check_cond0) % if s' is not terminal,            
            q_in=myState.Q(state_current,:);
            var_exp=exp(myParam.tau*q_in); % (N_actionx1)
            Prob_action=var_exp/sum(var_exp); % (N_actionx1)
            [bincounts] = histc(rand,[0 cumsum(Prob_action)]);
            action_hypo=find(bincounts(1:end-1)>0);            
            % filling in SARSA matrix - filling in (s',a')            last column means SARSA matrix has been completed.
            myState.SARSA(5:6)=myMap.RPE_SARSA( 2 * (myMap.trial - 1) + (myMap.index-1), 5:6 );
        else % if s' is terminal, then no decision has to be made, i.e., myState.SARSA(5)=0.
            % filling in SARSA matrix - filling in (s',a')            last column means SARSA matrix has been completed.
            myState.SARSA(5:6)=myMap.RPE_SARSA( 2 * (myMap.trial - 1) + (myMap.index-1), 5:6 ); % [0 1];
       end  
        
        
    %% SARSA - [update mode] reward prediction error & update Q
    case {'sarsa_update'}
        % ** myState.SARSA : (s,a,r,s',a')
        update_cond=(myState.SARSA(6)==1);
%         update_cond0=(myMap.IsTerminal(state_current)~=1);
        if(update_cond)
            % RPE
            if(myState.SARSA(5)~=0)
                myState.RPE_history(myState.index)=myState.SARSA(3)+...
                    myParam.gamma*myState.Q(myState.SARSA(4),myState.SARSA(5))-...
                    myState.Q(myState.SARSA(1),myState.SARSA(2));
            else % myState.SARSA(5)=0, i.e., terminal state s'
                myState.RPE_history(myState.index)=myState.SARSA(3)-...
                    myState.Q(myState.SARSA(1),myState.SARSA(2));
            end
            % Q-update
            myState.Q_old=myState.Q(myState.SARSA(1),myState.SARSA(2));
            myState.Q(myState.SARSA(1),myState.SARSA(2))=myState.Q(myState.SARSA(1),myState.SARSA(2))+...
                myParam.alpha*myState.RPE_history(myState.index);
        end
        % note: if s' is the terminal state ... no Q_sarsa(s';a') term in the sarsa update        
        % SARSA matrix reset
%         myState.SARSA=[0 0 0 0 0 0];
        
    
     %% FWD - [update mode] reward prediction error & update Q
    case {'fwd_update'}        
        % ** myState.SARSA : (s,a,r,s',0,0)
        update_cond=(myState.SARSA(4)~=0); % if state transition occurs,
%         update_cond0=(myMap.IsTerminal(state_current)~=1);
        if(update_cond)            
            % SPE
            myState.SPE_history(myState.index)=1-myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4));
            % T-update (increase T(s,a,s') and decrease T(s,a,-) to ensure the sum=1
            myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))=myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))+...
                myParam.alpha*myState.SPE_history(myState.index);
            array_rest=find([1:1:myState.N_state]~=myState.SARSA(4)); % the rest of the states
            for j=array_rest
                myState.T(myState.SARSA(1),myState.SARSA(2),j)=myState.T(myState.SARSA(1),myState.SARSA(2),j)*(1-myParam.alpha);
            end
            % Q-update            
            tmp_sum=0;
            tmp_sum=tmp_sum+myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))*...
                (myState.SARSA(3)+max(myState.Q(myState.SARSA(4),:))); % for (s,a,s')
            for j=array_rest % for the rest
                tmp_sum=tmp_sum+myState.T(myState.SARSA(1),myState.SARSA(2),j)*...
                    (myMap.reward(j)+max(myState.Q(j,:)));
            end
            myState.Q_old=myState.Q(myState.SARSA(1),myState.SARSA(2));
            myState.Q(myState.SARSA(1),myState.SARSA(2))=tmp_sum;
        end
        % note: if s' is the terminal state ... no Q_sarsa(s',a') term in the sarsa update
        % SARSA matrix reset
%         myState.SARSA=[0 0 0 0 0 0];

    case {'dpnmm_fwd_update'}        
        % ** myState.SARSA : (s,a,r,s',0,0)
        update_cond=(myState.SARSA(4)~=0); % if state transition occurs,
        %         update_cond0=(myMap.IsTerminal(state_current)~=1);
        if(update_cond)
            % SPE
            myState.T = myMap.SPE_T(:,:,:,2 * (myMap.trial - 1) + (myMap.index-1));
            myState.SPE_history(myState.index)=1-myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4));
            % T-update (increase T(s,a,s') and decrease T(s,a,-) to ensure the sum=1
            myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))=myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))+...
                myParam.alpha*myState.SPE_history(myState.index);
            array_rest=find([1:1:myState.N_state]~=myState.SARSA(4)); % the rest of the states
            for j=array_rest
                myState.T(myState.SARSA(1),myState.SARSA(2),j)=myState.T(myState.SARSA(1),myState.SARSA(2),j)*(1-myParam.alpha);
            end
            % Q-update            
            tmp_sum=0;
            tmp_sum=tmp_sum+myState.T(myState.SARSA(1),myState.SARSA(2),myState.SARSA(4))*...
                (myState.SARSA(3)+max(myState.Q(myState.SARSA(4),:))); % for (s,a,s')
            for j=array_rest % for the rest
                tmp_sum=tmp_sum+myState.T(myState.SARSA(1),myState.SARSA(2),j)*...
                    (myMap.reward(j)+max(myState.Q(j,:)));
            end
            myState.Q_old=myState.Q(myState.SARSA(1),myState.SARSA(2));
            myState.Q(myState.SARSA(1),myState.SARSA(2))=tmp_sum;
        end
        % note: if s' is the terminal state ... no Q_sarsa(s',a') term in the sarsa update
        % SARSA matrix reset
%         myState.SARSA=[0 0 0 0 0 0];




    %% BWD - [update mode] update the terminal Q if the goal is given
    case {'bwd_update'} % function [myState]=Model_RL(myState, myParam, myMap, mode, varargin)
        
%         ######### CHECK IF this works - 13 JUNE 2012 #################
            Q_bwd_before=myState.Q;
            % backward Q-update
            for ind_ll=[(max(myMap.level)-1):-1:1] %each level:: except the last level which is terminal
                state_ind_set=find(myMap.level==ind_ll);
                for state_ll=1:1:length(state_ind_set) %each state in each level
                    current_st=state_ind_set(state_ll);
                    for current_ac=1:1:myState.N_action %each action for each state
                        tmp_sum=0;                        
                        for j=[1:1:myState.N_state]
                            tmp_sum=tmp_sum+myState.T(current_st,current_ac,j)*...
                                (myMap.reward(j)+max(myState.Q(j,:)));
                        end
%                         tmp_sum=squeeze(myState.T(current_st,current_ac,:))'*(myMap.reward+max(myState.Q(j,:)));
                        myState.Q(current_st,current_ac)=tmp_sum;
                    end
                end
            end
            Q_bwd_after=myState.Q;
            myState.dQ=Q_bwd_after-Q_bwd_before;
            myState.dQ_bwd_energy=sqrt(sum(sum((myState.dQ).^2)));
            myState.dQ_mean_energy=mean(mean(myState.dQ));
            
            % after the bwd update, then set the flag 0
            myArbitrator_top.backward_flag=0;
        
        
    %% Wrong argument
    otherwise 
        disp('#### WARNING: check the mode name of the model function ####');
        
end

% myState.SARSA
end