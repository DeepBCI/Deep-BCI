function [myState myMap]=StateSpace_v1(myState,myMap,opt)
% input: myState (with current ".state_history" and ".action_history"
% parameter: myMap.action1, myMap.action2, myMap.reward
% output: myState (with updated ".index", ".state_history" and ".reward_history"

% opt.use_data=0; >> normal state transition
% opt.use_data=1; >> simply using myMap.data for state transition

%% readout
current_state=myState.state_history(myState.index);
current_action=myState.action_history(myState.index);


if(opt.use_data==0)
    if(myMap.IsTerminal(current_state)~=1)
        
        %% clock+1 (index synchronization)
        myMap.index=myMap.index+1; % 2,3
        myState.index=myMap.index;
        
        %% state transition
        prob_mat=myMap.action(1,current_action).prob(current_state,:); %(ex) [... 0 0.3 0 0.7]
        state_mat=myMap.action(1,current_action).connection(current_state,:); % (ex) [... 0 1 0 1]
        [tmp state_cand]=find(state_mat==1); % a set of candidate next-states
        if(rand<=prob_mat(state_cand(1)))
            % state transition
            myState.state_history(myState.index)=state_cand(1);
            % reward
            if(rand<=myMap.reward_prob(state_cand(1)))
                myState.reward_history(myState.index)=myMap.reward(state_cand(1));
            else
                myState.reward_history(myState.index)=0;
            end
        else
            % state transition
            myState.state_history(myState.index)=state_cand(2);
            % reward
            if(rand<=myMap.reward_prob(state_cand(2)))
                myState.reward_history(myState.index)=myMap.reward(state_cand(2));
            else
                myState.reward_history(myState.index)=0;
            end
        end
        
        %% filling in SARSA matrix - (r,s')
        myState.SARSA(3:4)=[myState.reward_history(myState.index) myState.state_history(myState.index)];
        
        %% If the new state is terminal,
        if(myMap.IsTerminal(myState.state_history(myState.index))==1)
            myState.JobComplete=1;
            myMap.JobComplete=1;
        end
        
    end
end



if(opt.use_data==1) % using saved behavior data for the state-transition
    if(myMap.IsTerminal(current_state)~=1)
        
        %% clock+1 (index synchronization)
        myMap.index=myMap.index+1; % 2,3
        myState.index=myMap.index;
        
        %% state transition                
        state_selected=myMap.data(myMap.trial,3+myMap.index);
        myState.state_history(myState.index)=state_selected;
        % reward
        myState.reward_history(myState.index)=myMap.data(myMap.trial,16);
                
        %% filling in SARSA matrix - (r,s')
        myState.SARSA(3:4)=[myState.reward_history(myState.index) myState.state_history(myState.index)];
        
        %% If the new state is terminal,
        if(myMap.IsTerminal(myState.state_history(myState.index))==1)
            myState.JobComplete=1;
            myMap.JobComplete=1;
        end
        
    end
end

end