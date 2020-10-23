%% 1. Parameter recovery analysis for behaviors
batch_parameter_recovery_behaviors()
%% 2. Parameter recovery analysis for simulation data
batch_parameter_recovery_simul(3)
batch_parameter_recovery_simul(4)
%% 3. Plot & save images  
tlist =[3,4];
names_= {'PFC-RL1','PFC-RL2'};
batch_parameter_recovery_correl(names_,tlist)