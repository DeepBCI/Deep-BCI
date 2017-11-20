function [myArbitrator myState1 myState2]=DirichletProcessNormalMix_Arbitration(myArbitrator, myState1, myState2, myMap)
% MAIN DESCRIPTION
% 
% Implemetation method #2 used, Rel model #1/#2/#3 should be implemented
% Read-out model : ArbModel = 1
% Poseterior stochastic model : ArbModel = 2
%% model option

% myArbitrator.opt_ArbModel=2; % 1: naive model(m1_wgt) . 2: posterior model(posterior) 




%% Preparation
myArbitrator.z_spe_flag = 0;
myArbitrator.z_rpe_flag = 0;


if(myArbitrator.ind_active_model==1)
    myState=myState1;
else
    myState=myState2;
end

%% index, state_history, action_history synchronization
% simply inherit because both should be in the same state
myArbitrator.index=myMap.index-1;
myArbitrator.state_history(myArbitrator.index)=myState.state_history(myArbitrator.index);
myArbitrator.action_history(myArbitrator.index)=myState.action_history(myArbitrator.index);

%% Reliabiltiy calculation, Inference based on ?
% posterior
myArbitrator.T_current1=min(myArbitrator.T_current1+1,myArbitrator.T); % update # of accumulated events

% 1. model 1 (m1)
% (0) backup old values
myArbitrator.m1_mean_old=myArbitrator.m1_mean;  myArbitrator.m1_var_old=myArbitrator.m1_var;    myArbitrator.m1_inv_Fano_old=myArbitrator.m1_inv_Fano;
% (1) find the corresponding row or POSTERIOR mode
if myArbitrator.opt_ArbModel == 1
    ind_update = myArbitrator.m1_CLUSTER_FOR_EACH_SESSION((myMap.trial-1)*2 + myArbitrator.index);
    if ind_update == myArbitrator.PE_index_m1(1)
        zspe = myState1.SPE_history(myMap.index);
        myArbitrator.zspe(1:end-1) = myArbitrator.zspe(2:end);
        myArbitrator.zspe(end) = zspe;
        myArbitrator.z_spe_flag = 1;
    end
elseif myArbitrator.opt_ArbModel == 2
    % (*) posterior of the cluster
    post_cluster_m1 = myArbitrator.mp_m1 .* normpdf(myState1.SPE_history(myState1.index), myArbitrator.model_m1_mean, sqrt(myArbitrator.model_m1_var))  / sum(myArbitrator.mp_m1 .* normpdf(myState1.SPE_history(myState1.index), myArbitrator.model_m1_mean, sqrt(myArbitrator.model_m1_var)));
    [ind_vec]=hist(rand, cumsum(post_cluster_m1));
    ind_update=find(ind_vec);
end
% (2) update the current column(=1) in PE_history
myArbitrator.m1_PE_history(:,2:end)=myArbitrator.m1_PE_history(:,1:end-1); % shift 1 column (toward past)
myArbitrator.m1_PE_history(:,1)=zeros(myArbitrator.K1,1); % empty the first column
myArbitrator.m1_PE_history(ind_update,1)=1; % add the count 1 in the first column
myArbitrator.m1_PE_num=myArbitrator.m1_PE_history*myArbitrator.discount_mat'; % compute discounted accumulated PE (default : do not discount)
% (3) posterior mean & var
sumK=sum(myArbitrator.m1_PE_num); % must be same as myArbitrator.T (time window length)
sumK_excl=sumK-myArbitrator.m1_PE_num;
myArbitrator.m1_mean=(1+myArbitrator.m1_PE_num)/(myArbitrator.K1+sumK);
myArbitrator.m1_var=((1+myArbitrator.m1_PE_num)/((myArbitrator.K1+sumK)^2))/(myArbitrator.K1+sumK+1).*(myArbitrator.K1+sumK_excl-1);
myArbitrator.m1_inv_Fano=myArbitrator.m1_mean./myArbitrator.m1_var;


% 2. model 2 (m2)
% (0) backup old values
myArbitrator.m2_mean_old=myArbitrator.m2_mean;  myArbitrator.m2_var_old=myArbitrator.m2_var;    myArbitrator.m2_inv_Fano_old=myArbitrator.m2_inv_Fano;
% (1) find the corresponding row or POSTERIOR mode
if myArbitrator.opt_ArbModel == 1
    ind_update = myArbitrator.m2_CLUSTER_FOR_EACH_SESSION((myMap.trial-1)*2 + myArbitrator.index);
    if ind_update == myArbitrator.PE_index_m2(1)
        zrpe = myState2.RPE_history(myMap.index);
        myArbitrator.zrpe(1:end-1) = myArbitrator.zrpe(2:end);
        myArbitrator.zrpe(end) = zrpe;
        myArbitrator.z_rpe_flag = 1;        
    end
elseif myArbitrator.opt_ArbModel == 2
    % (*) posterior of the cluster
    post_cluster_m2 = myArbitrator.mp_m2 .* normpdf(myState2.RPE_history(myState2.index), myArbitrator.model_m2_mean, sqrt(myArbitrator.model_m2_var))  / sum(myArbitrator.mp_m2 .* normpdf(myState2.RPE_history(myState2.index), myArbitrator.model_m2_mean, sqrt(myArbitrator.model_m2_var)));
    [ind_vec]=hist(rand, cumsum(post_cluster_m2));
    ind_update=find(ind_vec);
end
% (2) update the current column(=1) in PE_history
myArbitrator.m2_PE_history(:,2:end)=myArbitrator.m2_PE_history(:,1:end-1); % shift 1 column (toward past)
myArbitrator.m2_PE_history(:,1)=zeros(myArbitrator.K2,1); % empty the first column
myArbitrator.m2_PE_history(ind_update,1)=1; % add the count 1 in the first column
myArbitrator.m2_PE_num=myArbitrator.m2_PE_history*myArbitrator.discount_mat'; % compute discounted accumulated PE
% (3) posterior mean & var
sumK=sum(myArbitrator.m2_PE_num);
sumK_excl=sumK-myArbitrator.m2_PE_num;
myArbitrator.m2_mean=(1+myArbitrator.m2_PE_num)/(myArbitrator.K2+sumK);
myArbitrator.m2_var=((1+myArbitrator.m2_PE_num)/((myArbitrator.K2+sumK)^2))/(myArbitrator.K2+sumK+1).*(myArbitrator.K2+sumK_excl-1);
myArbitrator.m2_inv_Fano=myArbitrator.m2_mean./myArbitrator.m2_var;




%% Dynamic Arbitration

% if(myArbitrator.ind_active_model==1) % deleted [SW MAR 8]
%     input0=myArbitrator.m1_inv_Fano; % deleted [SW MAR 8]
%     input='*myArbitrator.m1_inv_Fano;   % deleted [SW MAR 8] 
% else % deleted [SW MAR 8]
%     input0=myArbitrator.m2_inv_Fano; % deleted [SW MAR 8]
%     input=myArbitrator.m2_wgt'*myArbitrator.m2_inv_Fano; % deleted [SW MAR 8]
% end % deleted [SW MAR 8]

myArbitrator.temp=0;%[myArbitrator.ind_active_model; input/sum(input0)];

input0=myArbitrator.m1_inv_Fano;  
if myArbitrator.opt_ArbModel == 1
    input1= myArbitrator.m1_wgt'*myArbitrator.m1_inv_Fano; % input1= [0 1 0]*myArbitrator.m1_inv_Fano;      % added [SW MAR 8]
elseif myArbitrator.opt_ArbModel == 2
    input1= post_cluster_m1 * myArbitrator.m1_inv_Fano; % should checked with calculation of 'Posterior of the cluster'
end
myArbitrator.transition_rate12=myArbitrator.A_12/(1+exp(myArbitrator.B_12*input1/sum(input0)));

input0=myArbitrator.m2_inv_Fano;  
if myArbitrator.opt_ArbModel == 1
    input2= myArbitrator.m2_wgt'*myArbitrator.m2_inv_Fano; % input1= [0 1 0]*myArbitrator.m1_inv_Fano;      % added [SW MAR 8]
elseif myArbitrator.opt_ArbModel == 2
    input2= post_cluster_m2 * myArbitrator.m2_inv_Fano; % should checked with calculation of 'Posterior of the cluster'
end
myArbitrator.transition_rate21=myArbitrator.A_21/(1+exp(myArbitrator.B_21*input2/sum(input0)));


myArbitrator.transition_rate12_prev=myArbitrator.transition_rate12;
myArbitrator.transition_rate21_prev=myArbitrator.transition_rate21;

myArbitrator.Tau= 1 / (myArbitrator.transition_rate12 + myArbitrator.transition_rate21); % alpha + beta term.
myArbitrator.m1_prob_inf=myArbitrator.transition_rate21*myArbitrator.Tau;
myArbitrator.m1_prob=myArbitrator.m1_prob_inf+(myArbitrator.m1_prob_prev-myArbitrator.m1_prob_inf)*exp((-1)*myArbitrator.Time_Step/myArbitrator.Tau);



% myArbitrator.opt_ArbModel = 0; % 2016 10 23 for working.
% switch myArbitrator.opt_ArbModel % bypassing dynamics
%     case 0 % full model
%         % as it is.
%     case 1 % invFano model   
%         myArbitrator.m1_prob=input1/(input1+input2);
%     case 2 % mean (of 0-PE) model
%         myArbitrator.m1_prob=myArbitrator.m1_mean(2)/(myArbitrator.m1_mean(2)+myArbitrator.m2_mean(2));        
%     case 3 % uncetainty model (NO USE)
% end

myArbitrator.m1_prob_prev=myArbitrator.m1_prob;
myArbitrator.m2_prob=1-myArbitrator.m1_prob;

%% choice of the model
myArbitrator.ind_active_model_prev=myArbitrator.ind_active_model;
if(myArbitrator.m1_prob>0.5)
    myArbitrator.ind_active_model=1;
    myArbitrator.num_m1_chosen=myArbitrator.num_m1_chosen+1;
    % there is no Q-value hand-over because sarsa computes Q based on SPE.
else
    myArbitrator.ind_active_model=2;    
    myArbitrator.num_m2_chosen=myArbitrator.num_m2_chosen+1;
    % Q-value hand-over : sarsa uses RPE-based Q only, does not use SPE.
%     if((myArbitrator.ind_active_model_prev==1)&&(myArbitrator.num_m2_chosen==0))
%         myState2.Q=myState1.Q;
%     end    
end


%% Q-value computation and action choice
% (1) Q-value computing
state_current=myArbitrator.state_history(myArbitrator.index);
action_current=myArbitrator.action_history(myArbitrator.index);
% [myArbitrator.ind_active_model myArbitrator.ind_active_model_prev myArbitrator.index state_current action_current]
% myArbitrator.Q(state_current,action_current)=...
%     ((myArbitrator.m1_prob*myState1.Q(state_current,action_current))^myArbitrator.p+...
%     (myArbitrator.m2_prob*myState2.Q(state_current,action_current))^myArbitrator.p)^(1/myArbitrator.p);
myArbitrator.Q=...
    ((myArbitrator.m1_prob*myState1.Q).^myArbitrator.p+...
    (myArbitrator.m2_prob*myState2.Q).^myArbitrator.p).^(1/myArbitrator.p);



% (2) softmax decision making
check_cond0=(myMap.IsTerminal(state_current)~=1);
if(check_cond0) % if not in a terminal state
    var_exp=exp(myArbitrator.tau_softmax*myArbitrator.Q(state_current,:)); % (N_actionx1)
    Prob_action=var_exp/sum(var_exp); % (N_actionx1)
    myArbitrator.action_prob(myArbitrator.index)=Prob_action(1);
    if(rand<Prob_action(1))
        myArbitrator.action_history(myArbitrator.index)=1;
    else
        myArbitrator.action_history(myArbitrator.index)=2;
    end
end

end