function [myArbitrator]=f_DirichletProcessNormalMix_Arbitration_Init(N_state,N_action,N_transition, myMap, data_in)
%% NAIVE model (distance based)
% weight 관련해서도 바꿔야함.

myArbitrator.T = 10;

myArbitrator.K1 = size(data_in{1,1}.DPNMM_MODEL.SPE, 1); % number of cluster of SPE
myArbitrator.K2 = size(data_in{1,1}.DPNMM_MODEL.RPE, 1); % number of cluster of RPE
 % Distance based cluster order ORDERING AND INDEX 이 과정 오래걸리니까 밖으로 빼서
 % DPNMM_MODEL 에 넣어줘서 m1TANK 대신에 가져오는게 낫지 않겠나. 일단 개발할때는 그대로 두자. 2016/11/20 DJ Kim.
 m1_mean = [];
 m2_mean = [];
 mp_m1 = [];
 mp_m2 = [];
 m1_var= [];
 m2_var= [];
 for i = 1 :  myArbitrator.K1 
     m1_mean = [m1_mean  mean(data_in{1, 1}.DPNMM_MODEL.SPE(i).CLUSTER_DATA)];
     mp_m1 = [mp_m1 size(data_in{1, 1}.DPNMM_MODEL.SPE(i).CLUSTER_DATA,2)];
     m1_var = [m1_var var(data_in{1, 1}.DPNMM_MODEL.SPE(i).CLUSTER_DATA)];
 end
 
 for i = 1 : myArbitrator.K2
     m2_mean = [m2_mean  mean(data_in{1, 1}.DPNMM_MODEL.RPE(i).CLUSTER_DATA)];
     mp_m2 = [mp_m2 size(data_in{1, 1}.DPNMM_MODEL.RPE(i).CLUSTER_DATA,2)];
     m2_var = [m2_var var(data_in{1, 1}.DPNMM_MODEL.RPE(i).CLUSTER_DATA)];
 end
 
 
 [~, myArbitrator.PE_index_m1] = sort(abs(m1_mean));
 [~, myArbitrator.PE_index_m2] = sort(abs(m2_mean));
% posterior 구하는 법은 적어놓은 메모를 통해서 또는 Bishop 432p 식을 통해서
% posterior 구하기 위해서는 각 클러스터의 mean, variance, mixing proposition을 넘겨주어야함.
% 여기에서 굳이 해서 넘겨줄 필요 없을 수도 이 m1_mean 등은 초기화 된 밸류로 넘겨줘야 나중에 bayesian
% arbitration 에서 쓸 수 가 있음. 그래서~ 바꿔둠
myArbitrator.model_m1_mean = m1_mean;
myArbitrator.model_m2_mean = m2_mean;
myArbitrator.model_m1_var = m1_var;
myArbitrator.model_m2_var = m2_var;
myArbitrator.mp_m1 = mp_m1/sum(mp_m1);
myArbitrator.mp_m2 = mp_m2/sum(mp_m2);
myArbitrator.T_current1=0; % # of accumulated events. cannot exceed T.
myArbitrator.T_current2=0; % # of accumulated events. cannot exceed T.

%% Which model you starts with
myArbitrator.ind_active_model=1;
myArbitrator.Time_Step=1.e-10; % the smaller, the slower

%% Bayesian part - PE history 
% row: first index = smallest PE -> last index - largest PE
% column: first index = most recent -> last index = most past
myArbitrator.m1_mean=1/3*ones(myArbitrator.K1,1); myArbitrator.m2_mean=1/3*ones(myArbitrator.K2,1);
myArbitrator.m1_var=(2/((3^2)*4))*ones(myArbitrator.K1,1);  myArbitrator.m2_var=(2/((3^2)*4))*ones(myArbitrator.K2,1);

myArbitrator.m1_PE_history=zeros(myArbitrator.K1,myArbitrator.T);  myArbitrator.m2_PE_history=zeros(myArbitrator.K2,myArbitrator.T);
myArbitrator.m1_PE_num=zeros(myArbitrator.K1,1);   myArbitrator.m2_PE_num=zeros(myArbitrator.K2,1);
myArbitrator.m1_inv_Fano=(myArbitrator.m1_mean)./myArbitrator.m1_var; myArbitrator.m2_inv_Fano=(myArbitrator.m2_mean)./myArbitrator.m2_var;
myArbitrator.discount_mat=ones(1,myArbitrator.T); % no discount

% for convenience, but should be formed in the MODELS making part. (It
% should be predefined as struct. 2016 11 21 DJ Kim
myArbitrator.m1_CLUSTER_FOR_EACH_SESSION = cell(size(data_in{1, 1}.HIST_behavior_info,2),1);
myArbitrator.m2_CLUSTER_FOR_EACH_SESSION = myArbitrator.m1_CLUSTER_FOR_EACH_SESSION;

teemp=[];
for i = 1 : size(data_in{1, 1}.HIST_behavior_info,2)
  teemp=[teemp size(data_in{1, 1}.HIST_behavior_info{i},1)];
end
teemp = cumsum(teemp)*2;
for i = 1 : size(data_in{1, 1}.HIST_behavior_info,2)
    if i == 1
        myArbitrator.m1_CLUSTER_FOR_EACH_SESSION{i,1} = data_in{1, 1}.DPNMM_MODEL.SPE(1).SAMPLED_CLASS(1: teemp(i));
        myArbitrator.m2_CLUSTER_FOR_EACH_SESSION{i,1} = data_in{1, 1}.DPNMM_MODEL.RPE(1).SAMPLED_CLASS(1: teemp(i));        
    else
        myArbitrator.m1_CLUSTER_FOR_EACH_SESSION{i,1} = data_in{1, 1}.DPNMM_MODEL.SPE(1).SAMPLED_CLASS(teemp(i-1)+1: teemp(i));        
        myArbitrator.m2_CLUSTER_FOR_EACH_SESSION{i,1} = data_in{1, 1}.DPNMM_MODEL.RPE(1).SAMPLED_CLASS(teemp(i-1)+1: teemp(i));        
    end
end




%% Dynamic arbitration part
% weights for {-PE(to be unlearned), 0PE, +PE(to be learned)}
myArbitrator.m1_wgt=zeros(myArbitrator.K1,1); myArbitrator.m1_wgt(myArbitrator.PE_index_m1(1)) = 1;
myArbitrator.m2_wgt=zeros(myArbitrator.K2,1); myArbitrator.m2_wgt(myArbitrator.PE_index_m2(1)) = 1;

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
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%% 여기부터는 Bayeian Arbitration Init 코드 보고 싶으면 원본 코드를 보던가 ctrl + T
%% create arbitrator + set all initial parameters
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
%%  1
% myArbitrator.K=3; %trichonomy of PE
% myArbitrator.T=10; %memory size = 10
% myArbitrator.T_half=6; %half-life % useledd in dpnmm 2016 11 20 by DJ Kim
% myArbitrator.T_current1=0; % # of accumulated events. cannot exceed T.
% myArbitrator.T_current2=0; % # of accumulated events. cannot exceed T.
% myArbitrator.PE_tolerance_m1=0.5; % defines threshold for zero PE
% myArbitrator.PE_tolerance_m2=9.0; % defines threshold for zero PE
% 
% %% Which model you starts with
% myArbitrator.ind_active_model=1;
% % [NOTE] if you want to run only ".ind_active_model", set ".Time_Step" extremely slower.
% myArbitrator.Time_Step=1.e-10; % the smaller, the slower
% 
% myArbitrator.m1_thr_PE=myArbitrator.PE_tolerance_m1*[-1 1]; myArbitrator.m2_thr_PE=myArbitrator.PE_tolerance_m2*[-1 1];% length = myArbitrator.K-1
% 
% %% Bayesian part - PE history 
% % row: first index = smallest PE -> last index - largest PE
% % column: first index = most recent -> last index = most past
% myArbitrator.m1_PE_history=zeros(myArbitrator.K,myArbitrator.T);  myArbitrator.m2_PE_history=zeros(myArbitrator.K,myArbitrator.T);
% myArbitrator.m1_PE_num=zeros(myArbitrator.K,1);   myArbitrator.m2_PE_num=zeros(myArbitrator.K,1);
% 
% % myArbitrator.discount_mat=exp((-1)*log(2)*([0:1:myArbitrator.T-1]/(myArbitrator.T_half-1))); % prepare discount mat: 1xT
% myArbitrator.discount_mat=ones(1,myArbitrator.T); % no discount
% 
% %% Bayesian part - m2 RPE estimator (Jan25, 2013)
% myArbitrator.m2_absPEestimate_lr=0.1;
% myArbitrator.m2_absPEestimate=0.0;
% 
% %% Bayesian part - mean,var,inv_Fano
% myArbitrator.m1_mean=1/3*ones(myArbitrator.K,1); myArbitrator.m2_mean=1/3*ones(myArbitrator.K,1);
% myArbitrator.m1_var=(2/((3^2)*4))*ones(myArbitrator.K,1);  myArbitrator.m2_var=(2/((3^2)*4))*ones(myArbitrator.K,1);
% myArbitrator.m1_inv_Fano=(myArbitrator.m1_mean)./myArbitrator.m1_var; myArbitrator.m2_inv_Fano=(myArbitrator.m2_mean)./myArbitrator.m2_var;
% 
% %% Dynamic arbitration part
% % weights for {-PE(to be unlearned), 0PE, +PE(to be learned)}
% myArbitrator.m1_wgt=[-0.2 0.7 -0.2]'; myArbitrator.m2_wgt=myArbitrator.m1_wgt;
% % myArbitrator.m1_wgt=[0 1.0 0]'; myArbitrator.m2_wgt=myArbitrator.m1_wgt;
% % first, try to visualize the function with "SIMUL_Disp_gatingFn.m"
% % and then use those parameter values
% myArbitrator.A_12=1.0;  myArbitrator.B_12=.2e1;
% myArbitrator.A_21=1.2;  myArbitrator.B_21=1.0e1;
% % myArbitrator.A_21=1.0;  myArbitrator.B_21=1.5e-2;
% if(myArbitrator.ind_active_model==1)
%     myArbitrator.m1_prob_prev=0.7; % do not use 0.5 which causes oscillation.
% else
%     myArbitrator.m1_prob_prev=0.3;
% end
% myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
% myArbitrator.m1_prob=myArbitrator.m1_prob_prev;
% myArbitrator.m2_prob=myArbitrator.m2_prob_prev;
% 
% 
% %% Q-integration part
% myArbitrator.p=1; % 1:expectation, 1e1:winner-take-all
% myArbitrator.tau_softmax=0.5; % use the same value as sarsa/fwd
% myArbitrator.action_history=zeros(N_transition,1);
% myArbitrator.action_prob=zeros(N_transition,1);
% myArbitrator.Q=zeros(N_state,N_action);
% myArbitrator.state_history=zeros(N_transition,1);    myArbitrator.state_history(1,1)=1;
% myArbitrator.action_history=zeros(N_transition,1);
% 
% % set the first transition rate to be the equilibrium rate
% if(myArbitrator.B_21~=myArbitrator.B_12)
%         myArbitrator.inv_Fano_equilibrium=log(myArbitrator.A_21/myArbitrator.A_12)/(myArbitrator.B_21-myArbitrator.B_12); % applied only for the unnormalized case
% %     myArbitrator.inv_Fano_equilibrium=.5;
%     myArbitrator.transition_rate12_prev=myArbitrator.A_12*exp((-1)*myArbitrator.B_12*myArbitrator.inv_Fano_equilibrium);
%     myArbitrator.transition_rate21_prev=myArbitrator.transition_rate12_prev;
% else
%     %     myArbitrator.inv_Fano_equilibrium=150;
%     myArbitrator.inv_Fano_equilibrium=.1;
%     %     myArbitrator.transition_rate12_prev=myArbitrator.A_12*exp((-1)*myArbitrator.B_12*myArbitrator.inv_Fano_equilibrium);
%     myArbitrator.transition_rate12_prev=myArbitrator.inv_Fano_equilibrium;
%     myArbitrator.transition_rate21_prev=myArbitrator.transition_rate12_prev;
% end
% myArbitrator.transition_rate12=myArbitrator.transition_rate12_prev;
% myArbitrator.transition_rate21=myArbitrator.transition_rate21_prev;
% 
% %% MISC
% if(myArbitrator.ind_active_model==1)
%     myArbitrator.num_m1_chosen=1;
%     myArbitrator.num_m2_chosen=0;
% else
%     myArbitrator.num_m1_chosen=0;
%     myArbitrator.num_m2_chosen=1;
% end
% 
% myArbitrator.INPUT0=[];
% myArbitrator.INPUT1=[];
% myArbitrator.INPUT2=[];


   

end