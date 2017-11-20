function [MODEL]=Init_DirichletProcessNormalMix_Arbitration_Init(PE,alpha)
    addpath([pwd '/DPNMM/']);
    
    num_sweeps = 1000;
    mu_0 = zeros(1,1);
    k_0 = .05;
    lambda_0 = eye(1)*.02;
    v_0 = 1+1;
    a_0 = 1;
    b_0 = 1;
    alpha_0 = alpha; 
    
    
    %% SPE
%     disp('@@@@@@@@@@@@@@@@ SPE update');
    in_sample_training_data = PE.SPE(2,:);
    [class_id_samples, num_classes_per_sample, model_scores, alpha_record] = collapsed_gibbs_sampler(...
        in_sample_training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0, alpha_0);
    MAX=max(class_id_samples(:,num_sweeps));
    Y=cell(MAX,1);
    x=PE.SPE(2,:);
    for i = 1 : PE.trials
        Y{class_id_samples(i,num_sweeps)}=[Y{class_id_samples(i,num_sweeps)} x(i)];
    end
    SPE = struct('SAMPLED_CLASS', class_id_samples(:,num_sweeps), 'CLUSTER_DATA', Y, 'NUMBER_OF_CLUSTER',num_classes_per_sample(num_sweeps), 'MODEL_SCORE',model_scores, 'ALPHA_RECORD',alpha_record) ;
    
    
    %% RPE
%     disp('@@@@@@@@@@@@@@@@ RPE update');    
    in_sample_training_data = PE.RPE(2,:);
    [class_id_samples, num_classes_per_sample, model_scores, alpha_record] = collapsed_gibbs_sampler(...
        in_sample_training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0, alpha_0);
    MAX=max(class_id_samples(:,num_sweeps));
    Y=cell(MAX,1);
    x=PE.RPE(2,:);
    for i = 1 : PE.trials
        Y{class_id_samples(i,num_sweeps)}=[Y{class_id_samples(i,num_sweeps)} x(i)];
    end
    RPE = struct('SAMPLED_CLASS', class_id_samples(:,num_sweeps), 'CLUSTER_DATA', Y, 'NUMBER_OF_CLUSTER',num_classes_per_sample(num_sweeps), 'MODEL_SCORE',model_scores, 'ALPHA_RECORD',alpha_record) ;
    
    
    %% TOTAL 
    MODEL = struct('SPE',SPE,'RPE',RPE);

end