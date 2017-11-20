function [ret_particles, ret_weights, ret_K_plus] = particle_filter(training_data, ...
    num_particles, a_0, b_0, mu_0, k_0, v_0, ...
    lambda_0,alpha, partial_labels)
%
% function [ret_particles, ret_weights, ret_K_plus] =
%     particle_filter(training_data, ...
%     num_particles, a_0, b_0, mu_0, k_0, v_0, ...
%     lambda_0,alpha)
%
% This function is written to illustrate online posterior inference
% in an IGMM, of course in a real "online" setting one wouldn't have access
% to all the data upfront.  This code (more generally this technique) is
% factored so that it should be very easy to refactor the code so that the
% data is processed one observation at a time and the particle set is
% also returned having observed each observation

pc_max_ind = 1e5;
pc_gammaln_by_2 = 1:pc_max_ind;
pc_gammaln_by_2 = gammaln(pc_gammaln_by_2/2);
pc_log_pi = reallog(pi);
pc_log = reallog(1:pc_max_ind);

class_id_type = 'uint8';
max_class_id = intmax(class_id_type);

Y = training_data;
T = size(Y,2);
D = size(Y,1);
N = num_particles;

% intialize space for the particles and weights
particles = zeros(N,T,2,class_id_type);
weights = zeros(N,2);
% K_plus will be the numebr of classes currently in each particle
K_plus = ones(N,2,class_id_type);

% pre-allocate space for per-particle sufficient statistics and other
% efficiency related variables
means = zeros(D,max_class_id,N,2);
sum_squares = zeros(D,D,max_class_id,N,2);
inv_cov = zeros(D,D,max_class_id,N,2);
log_det_cov = zeros(max_class_id,N,2);
counts = zeros(max_class_id,N,2,'uint32');

% cp is the set of current particles
cp =1;

if nargin < 10

    index_at_which_to_start_estimation = 2;

    % seat the first customer at the first table in all particles
    particles(:,1,cp) = 1;
    weights(:,cp) = 1/N;

    % initialize the partial sums for seqential covariance and mean updates and
    % precompute the covariance of the posterior predictive student-T with one
    % observation
    y = training_data(:,1);
    yyT = y * y';
    [lp ldc ic] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,1,y,yyT,k_0,mu_0,v_0,lambda_0);

    for i=1:N
        means(:,1,i,cp) = y;
        sum_squares(:,:,1,i,cp) = yyT;
        counts(1,i,cp) = 1;
        log_det_cov(1,i,cp) = ldc;
        inv_cov(:,:,1,i,cp) = ic;
    end


else
    if size(partial_labels,2)==1
        partial_labels = repmat(partial_labels',N,1);
    else
        partial_labels = partial_labels';
    end
    num_already_seated = size(partial_labels,2);
    index_at_which_to_start_estimation = num_already_seated+1;
    for i = 1:N

        particles(i,1:num_already_seated,cp) = partial_labels(i,:); % can be smarter here about integrated MCMC samples -- i.e. more than one
        weights(:,cp) = 1/N;


        num_tables = length(unique(partial_labels(i,:)));
        K_plus(:,cp) = num_tables;

        for k = 1:num_tables
            class_k_inds = partial_labels(i,:) == k;
            Y = training_data(:,class_k_inds);
            mean_Y = mean(Y,2);

            SS2 = zeros(size(Y,1),size(Y,1));
            for z = 1: sum(class_k_inds)
                SS2 = SS2 + Y(:,z)*Y(:,z)';
            end

            [lp ldc ic] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,Y(:,1),1,Y(:,1),SS2,k_0,mu_0,v_0,lambda_0);

            %         for i = 1:N
            means(:,k,i,cp) = mean_Y;
            sum_squares(:,:,k,i,cp) = SS2;
            counts(k,i,cp) = sum(class_k_inds);
            log_det_cov(k,i,cp) = ldc;
            inv_cov(:,:,k,i,cp) = ic;
            %         end
        end
    end

end

% initialize timers
time_1_obs = 0;
total_time = 0;

% initialize sum_K_plus which is the sum of the total number of classes in
% each particle


% go through observations 2 through T sequentially
for t = index_at_which_to_start_estimation:T
    sum_K_plus = sum(K_plus(:,cp));
    % we need this for memory management later
    max_K_plus = uint32(max(K_plus(:,cp)))+1;
    %     disp([' sum K^+: ' num2str(sum(K_plus(cp,:))) ', max K^+ ' num2str(max_K_plus)]);
    y = training_data(:,t);

    E_K_plus = sum_K_plus/N;

    total_time = total_time + time_1_obs;
    if t==2
        disp(['CRP PF:: Obs: ' num2str(t) '/' num2str(T) ]);
    elseif mod(t,5)==0
        rem_time = (time_1_obs*.05 + 0.95*(total_time/t))*T-total_time;
        if rem_time < 0
            rem_time = 0;
        end
        disp(['CRP PF:: Obs: ' num2str(t) '/' num2str(T) ', Rem. Time: '...
            secs2hmsstr(rem_time) ', Ave. Time: ' ...
            secs2hmsstr((total_time/(t-2))) ', Elaps. Time: ' ...
            secs2hmsstr(total_time) ', E[K^+] ' num2str(E_K_plus)]);
    end
    tic


    % M == the number of putative particles to generate
    M = sum_K_plus+N;
    putative_particles = zeros(2,M, 'uint32');
    putative_weights = ones(1,M);


    % si and ei are the starting and ending indexes of the K_plus(n)+1
    % distinct putative particles that are generated from particle n
    si = 1;
    ei = uint32(K_plus(1,cp)+1);

    putative_pf = exp(lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,0,[],[],k_0,mu_0,v_0,lambda_0));

    for n = 1:N
        num_options = ei-si+1; % i.e. K_plus(1)+1
        if num_options > max_class_id
            error(['K^plus has exceeded the maximum value of ' class_id_type ])
        end

        % copy the old part and choose each table once for the new parts
        putative_particles(1,si:ei) = n;
        putative_particles(2,si:ei) = 1:(num_options);

        % calculate the probability of each new putative particle under the
        % CRP prior alone
        m_k = double(counts(1:K_plus(n,cp),n,cp))';

        prior = [m_k alpha];
        prior = prior./(t+alpha-1);

        % update the weights so that the particles (and weights) now represent the
        % predictive distribution
        putative_weights(si:ei) = weights(n,cp)*prior;

        % update the weights so that the particles (and weights) now
        % represent the posterior distribution at ''timestep'' t
        posterior_predictive_p = zeros(size(prior));
        for pnp_id = 1: num_options-1;
            lpf = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,double(counts(pnp_id,n,cp)),means(:,pnp_id,n,cp),sum_squares(:,:,pnp_id,n,cp),k_0,mu_0,v_0,lambda_0,log_det_cov(pnp_id,n,cp),inv_cov(:,:,pnp_id,n,cp));

            posterior_predictive_p(pnp_id) = exp(lpf);
        end
        posterior_predictive_p(end) = putative_pf;
        putative_weights(si:ei) = putative_weights(si:ei).*posterior_predictive_p;

        % maintain indexing for putative particle placement in large array
        si = ei+1;
        if n~=N
            ei = si+uint32(K_plus(n+1,cp));
        end
    end

    % the M weights are computed up to a proportionality so we normalize
    % them here
    putative_weights = putative_weights ./ sum(putative_weights);

    c = find_optimal_c(putative_weights,N);

    % find pass-through ratio
    pass_inds = putative_weights>1/c;
    num_pass = sum(pass_inds);

    if cp == 1
        np = 2;
    else
        np = 1;
    end

    yyT = y*y';



    if num_pass >0
        particles(1:num_pass,1:t-1,np) = particles(putative_particles(1,pass_inds),1:t-1,cp);

        particles(1:num_pass,t,np) = putative_particles(2,pass_inds);

        weights(1:num_pass,np) = putative_weights(pass_inds);


        passing_class_id_ys = putative_particles(2,pass_inds);
        passing_orig_partical_ids = putative_particles(1,pass_inds);
        for npind = 1:num_pass
            class_id_y = passing_class_id_ys(npind);
            originating_particle_id = passing_orig_partical_ids(npind);
            originating_particle_K_plus = K_plus(originating_particle_id,cp);

            new_count = counts(class_id_y,originating_particle_id,cp)+1;


            if new_count == 1
                K_plus(npind,np) = originating_particle_K_plus+1;

            else
                K_plus(npind,np) = originating_particle_K_plus;

            end

            counts(1:max_K_plus,npind,np) = counts(1:max_K_plus,originating_particle_id,cp);
            counts(class_id_y,npind,np) = new_count;

            old_mean = means(:,class_id_y,originating_particle_id,cp);
            means(:,1:max_K_plus,npind,np) = means(:,1:max_K_plus,originating_particle_id,cp); %check this one too
            means(:,class_id_y,npind,np) =  old_mean + (1/double(new_count)) * (y - old_mean);
            sum_squares(:,:,1:max_K_plus,npind,np) = sum_squares(:,:,1:max_K_plus,originating_particle_id,cp); % check this line
            sum_squares(:,:,class_id_y,npind,np) = sum_squares(:,:,class_id_y,originating_particle_id,cp) + yyT;

            % here we use a hidden feature of  lp_tpp_helper  in that
            % it will calculate the log_det_cov and the inv_cov for us
            % automatically.  we don't care about lp here at all
            [lp ldc ic] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,double(new_count),means(:,class_id_y,npind,np),sum_squares(:,:,class_id_y,npind,np),k_0,mu_0,v_0,lambda_0);


            log_det_cov(1:max_K_plus,npind,np) = log_det_cov(1:max_K_plus,originating_particle_id,cp);

            log_det_cov(class_id_y,npind,np) = ldc;

            inv_cov(:,:,1:max_K_plus,npind,np) = inv_cov(:,:,1:max_K_plus,originating_particle_id,cp); % and this one

            inv_cov(:,:,class_id_y,npind,np) = ic;

        end

    end

    if N-num_pass > 0
        weights(num_pass+1:end,np) = 1/c;

        picked_putative_particles = stratified_resample(putative_particles(:,~pass_inds),putative_weights(~pass_inds),N-num_pass);
        npind = num_pass +1;
        for ppind = 1:N-num_pass
            class_id_y = picked_putative_particles(2,ppind);
            originating_particle_id = picked_putative_particles(1,ppind);
            originating_particle_K_plus = K_plus(originating_particle_id,cp);
            %             particles(npind,1:t,np) = [reshape(particles(originating_particle_id,1:t-1,cp),1,t-1) class_id_y];
            particles(npind,1:t,np) = [particles(originating_particle_id,1:t-1,cp) class_id_y];
            new_count = counts(class_id_y,originating_particle_id,cp)+1;
            K_plus(npind,np) = originating_particle_K_plus;
            if new_count == 1
                K_plus(npind,np) = originating_particle_K_plus+1;
            end

            counts(1:max_K_plus,npind,np) = counts(1:max_K_plus,originating_particle_id,cp);
            counts(class_id_y,npind,np) = new_count;

            old_mean = means(:,class_id_y,originating_particle_id,cp);

            means(:,1:max_K_plus,npind,np) = means(:,1:max_K_plus,originating_particle_id,cp); % check this line
            means(:,class_id_y,npind,np) =  old_mean + (1/double(new_count)) * (y - old_mean);

            sum_squares(:,:,1:max_K_plus,npind,np) = sum_squares(:,:,1:max_K_plus,originating_particle_id,cp);
            sum_squares(:,:,class_id_y,npind,np) = sum_squares(:,:,class_id_y,originating_particle_id,cp) + yyT;

            % here we use a hidden feature of  lp_tpp_helper  in that
            % it will calculate the log_det_cov and the inv_cov for us
            % automatically.  we don't care about lp here at all
            [lp ldc ic] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,double(new_count),means(:,class_id_y,npind,np),sum_squares(:,:,class_id_y,npind,np),k_0,mu_0,v_0,lambda_0);

            log_det_cov(1:max_K_plus,npind,np) = log_det_cov(1:max_K_plus,originating_particle_id,cp);
            log_det_cov(class_id_y,npind,np) = ldc;
            inv_cov(:,:,1:max_K_plus,npind,np) = inv_cov(:,:,1:max_K_plus,originating_particle_id,cp);
            inv_cov(:,:,class_id_y,npind,np) = ic;
            npind = npind +1;
        end


    end

    cp = np;
    time_1_obs = toc;
end
munlock('lp_tpp_helper')
ret_particles = squeeze(particles(:,:,cp));
ret_weights = squeeze(weights(:,cp));
ret_K_plus = squeeze(K_plus(:,cp));