function [myState]=Model_RL_Init(N_state,N_action,N_transition)

myState.N_state=N_state;
myState.N_action=N_action;
myState.N_transition=N_transition;
myState.index=1; % current index
myState.state_history=zeros(N_transition,1);    myState.state_history(1,1)=1;
myState.action_history=zeros(N_transition,1);
myState.action_prob=zeros(N_transition,1);
myState.reward_history=zeros(N_transition,1);
myState.RPE_history=zeros(N_transition,1);
myState.SPE_history=zeros(N_transition,1);
myState.Q=zeros(N_state,N_action);
% myState.IsTerminal=zeros(N_state,1);    myState.IsTerminal(6:21)=1;
myState.JobComplete=0; % if the current state is final, then 1.
% for SARSA model
myState.SARSA=zeros(1,6); % (s,a,r,s',a',complete_index)
% for FWD model
myState.T=zeros(N_state,2,N_state); % (s,a,s')
for j=1:1:N_state % initialized to uniform distributions
    for k=1:1:2
        myState.T(j,k,:)=1/N_state;
    end
end

end