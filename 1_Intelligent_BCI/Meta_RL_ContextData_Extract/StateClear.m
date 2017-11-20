function [myState myMap]=StateClear(myState, myMap)

N=length(myState.state_history);

myState.index=1; % current index
myState.state_history=zeros(N,1);    myState.state_history(1,1)=1;
myState.action_history=zeros(N,1);
myState.reward_history=zeros(N,1);
myState.RPE_history=zeros(N,1);
myState.SPE_history=zeros(N,1);
myState.SARSA=[0 0 0 0 0 0];
myState.JobComplete=0;

myMap.index=1;
myMap.JobComplete=0;

end