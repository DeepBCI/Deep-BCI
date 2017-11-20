function [MODEL]=Init_DirichletProcessNormalMix_Arbitration_Init2(PE,alpha)
    addpath([pwd '\DPNMM\']);
    
    num_sweeps = 1000;
    mu_0 = zeros(1,1);
    k_0 = .05;
    lambda_0 = eye(1)*.02;
    v_0 = 1+1;
    a_0 = 1;
    b_0 = 1;
    alpha_0 = alpha; 
    
    
    %% SPE
    disp('@@@@@@@@@@@@@@@@ SPE update');
    in_sample_training_data = PE.SPE(2,:);
    
    [class_id_samples, num_classes_per_sample, model_scores, alpha_record] = collapsed_gibbs_sampler(...
        in_sample_training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0, alpha_0);
    
    burned_in_index = 51;
    num_particles = num_sweeps-burned_in_index+1;
    [PE_sortings, PE_sorting_weights, number_of_clusters_in_each_sorting] = particle_filter(in_sample_training_data, ...
        num_particles, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0,mean(alpha_record), class_id_samples);
    
    PE_sortings=PE_sortings';

    MAX=max(PE_sortings(:,num_particles));
    Y=cell(MAX,1);
    x=PE.SPE(2,:);
    
    for i = 1 : PE.trials
        Y{PE_sortings(i,num_particles)}=[Y{PE_sortings(i,num_particles)} x(i)];
    end
    SPE = struct('SAMPLED_CLASS', PE_sortings(:,num_particles), 'CLUSTER_DATA', Y, 'NUMBER_OF_CLUSTER',number_of_clusters_in_each_sorting(num_particles) ,'ALPHA_USED',mean(alpha_record)) ;
    
    
    %% RPE
    disp('@@@@@@@@@@@@@@@@ RPE update');    
    in_sample_training_data = PE.RPE(2,:);
    
    [class_id_samples, num_classes_per_sample, model_scores, alpha_record] = collapsed_gibbs_sampler(...
        in_sample_training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0, alpha_0);
    
    burned_in_index = 51;
    num_particles = num_sweeps-burned_in_index+1;
    [PE_sortings, PE_sorting_weights, number_of_clusters_in_each_sorting] = particle_filter(in_sample_training_data, ...
        num_particles, a_0, b_0, mu_0, k_0, v_0, ...
        lambda_0,mean(alpha_record), class_id_samples);
    
    PE_sortings=PE_sortings';
        
    MAX=max(PE_sortings(:,num_particles));
    Y=cell(MAX,1);
    x=PE.RPE(2,:);
    
    for i = 1 : PE.trials
        Y{PE_sortings(i,num_particles)}=[Y{PE_sortings(i,num_particles)} x(i)];
    end
    RPE = struct('SAMPLED_CLASS', PE_sortings(:,num_particles), 'CLUSTER_DATA', Y, 'NUMBER_OF_CLUSTER',number_of_clusters_in_each_sorting(num_particles) ,'ALPHA_USED',mean(alpha_record)) ;
    
    
    %% TOTAL 
    MODEL = struct('SPE',SPE,'RPE',RPE);

end