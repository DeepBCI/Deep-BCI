function [myArbitrator]=Bayesian_Arbitration_Init(N_state,N_action,N_transition)

%% create arbitrator + set all initial parameters

myArbitrator.K=3; %trichonomy of PE
myArbitrator.T=10; %memory size = 10
myArbitrator.T_half=6; %half-life
myArbitrator.T_current1=0; % # of accumulated events. cannot exceed T.
myArbitrator.T_current2=0; % # of accumulated events. cannot exceed T.
myArbitrator.PE_tolerance_m1=0.5; % defines threshold for zero PE
myArbitrator.PE_tolerance_m2=9.0; % defines threshold for zero PE

%% Which model you starts with
myArbitrator.ind_active_model=1;
% [NOTE] if you want to run only ".ind_active_model", set ".Time_Step" extremely slower.
myArbitrator.Time_Step=1.e-10; % the smaller, the slower

myArbitrator.m1_thr_PE=myArbitrator.PE_tolerance_m1*[-1 1]; myArbitrator.m2_thr_PE=myArbitrator.PE_tolerance_m2*[-1 1];% length = myArbitrator.K-1

%% Bayesian part - PE history 
% row: first index = smallest PE -> last index - largest PE
% column: first index = most recent -> last index = most past
myArbitrator.m1_PE_history=zeros(myArbitrator.K,myArbitrator.T);  myArbitrator.m2_PE_history=zeros(myArbitrator.K,myArbitrator.T);
myArbitrator.m1_PE_num=zeros(myArbitrator.K,1);   myArbitrator.m2_PE_num=zeros(myArbitrator.K,1);

% myArbitrator.discount_mat=exp((-1)*log(2)*([0:1:myArbitrator.T-1]/(myArbitrator.T_half-1))); % prepare discount mat: 1xT
myArbitrator.discount_mat=ones(1,myArbitrator.T); % no discount

%% Bayesian part - m2 RPE estimator (Jan25, 2013)
myArbitrator.m2_absPEestimate_lr=0.1;
myArbitrator.m2_absPEestimate=0.0;

%% Bayesian part - mean,var,inv_Fano
myArbitrator.m1_mean=1/3*ones(myArbitrator.K,1); myArbitrator.m2_mean=1/3*ones(myArbitrator.K,1);
myArbitrator.m1_var=(2/((3^2)*4))*ones(myArbitrator.K,1);  myArbitrator.m2_var=(2/((3^2)*4))*ones(myArbitrator.K,1);
myArbitrator.m1_inv_Fano=(myArbitrator.m1_mean)./myArbitrator.m1_var; myArbitrator.m2_inv_Fano=(myArbitrator.m2_mean)./myArbitrator.m2_var;

%% Dynamic arbitration part
% weights for {-PE(to be unlearned), 0PE, +PE(to be learned)}
myArbitrator.m1_wgt=[-0.2 0.7 -0.2]'; myArbitrator.m2_wgt=myArbitrator.m1_wgt;
% myArbitrator.m1_wgt=[0 1.0 0]'; myArbitrator.m2_wgt=myArbitrator.m1_wgt;
% first, try to visualize the function with "SIMUL_Disp_gatingFn.m"
% and then use those parameter values
myArbitrator.A_12=1.0;  myArbitrator.B_12=.2e1;
myArbitrator.A_21=1.2;  myArbitrator.B_21=1.0e1;
% myArbitrator.A_21=1.0;  myArbitrator.B_21=1.5e-2;
if(myArbitrator.ind_active_model==1)
    myArbitrator.m1_prob_prev=0.7; % do not use 0.5 which causes oscillation.
else
    myArbitrator.m1_prob_prev=0.3;
end
myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
myArbitrator.m1_prob=myArbitrator.m1_prob_prev;
myArbitrator.m2_prob=myArbitrator.m2_prob_prev;


%% Q-integration part
myArbitrator.p=1; % 1:expectation, 1e1:winner-take-all
myArbitrator.tau_softmax=0.5; % use the same value as sarsa/fwd
myArbitrator.action_history=zeros(N_transition,1);
myArbitrator.action_prob=zeros(N_transition,1);
myArbitrator.Q=zeros(N_state,N_action);
myArbitrator.state_history=zeros(N_transition,1);    myArbitrator.state_history(1,1)=1;
myArbitrator.action_history=zeros(N_transition,1);

% set the first transition rate to be the equilibrium rate
if(myArbitrator.B_21~=myArbitrator.B_12)
        myArbitrator.inv_Fano_equilibrium=log(myArbitrator.A_21/myArbitrator.A_12)/(myArbitrator.B_21-myArbitrator.B_12); % applied only for the unnormalized case
%     myArbitrator.inv_Fano_equilibrium=.5;
    myArbitrator.transition_rate12_prev=myArbitrator.A_12*exp((-1)*myArbitrator.B_12*myArbitrator.inv_Fano_equilibrium);
    myArbitrator.transition_rate21_prev=myArbitrator.transition_rate12_prev;
else
    %     myArbitrator.inv_Fano_equilibrium=150;
    myArbitrator.inv_Fano_equilibrium=.1;
    %     myArbitrator.transition_rate12_prev=myArbitrator.A_12*exp((-1)*myArbitrator.B_12*myArbitrator.inv_Fano_equilibrium);
    myArbitrator.transition_rate12_prev=myArbitrator.inv_Fano_equilibrium;
    myArbitrator.transition_rate21_prev=myArbitrator.transition_rate12_prev;
end
myArbitrator.transition_rate12=myArbitrator.transition_rate12_prev;
myArbitrator.transition_rate21=myArbitrator.transition_rate21_prev;

%% MISC
if(myArbitrator.ind_active_model==1)
    myArbitrator.num_m1_chosen=1;
    myArbitrator.num_m2_chosen=0;
else
    myArbitrator.num_m1_chosen=0;
    myArbitrator.num_m2_chosen=1;
end

myArbitrator.INPUT0=[];
myArbitrator.INPUT1=[];
myArbitrator.INPUT2=[];





%% 2016 11 06 for working.

end