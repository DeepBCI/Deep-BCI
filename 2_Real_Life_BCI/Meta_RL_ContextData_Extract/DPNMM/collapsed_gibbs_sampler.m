function [class_id, K_record, lp_record, alpha_record] = collapsed_gibbs_sampler(...
    training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0, ...
    lambda_0, alpha_0, trace_plot_number, progress_plot_number, trace_movie_file, progress_movie_file)
% function [class_id, K_record, lp_record, alpha_record] = collapsed_gibbs_sampler(
%           training_data, num_sweeps, a_0, b_0, mu_0, k_0, v_0,
%                                   lambda_0, alpha_0, trace_plot_number,
%                                   progress_plot_number, trace_movie_file, progress_movie_file)
%
%   Generates samples from the infinite Gaussian mixture model posterior.
% The model parameter samples returned are the class_id's assigned to each
% data point.  Also returned are K_record, lp_record, and alpha_record
% which are the the record of K^+, the number of mixture densities, the log
% probability of the training data under the model, and the distribution
% over the Dirichlet hyperparameter alpha.
%
%  Input variables are the dxn training_data (d=dim, n=#points), the number
% of sampler sweeps, the gamma prior on alpha parameters a_0 and b_0, and
% the invesh-wishart/normal hyperparameters mu_0, k_0, v_0, lambda_0
% which conform to Gelman's notation. alpha_0 gives you the option of
% specifying the intial value of alpha used by the algorithm (defaults to
% 1) If trace_plot_number and progress_plot_number two plots will be
% generated as the sampler runs, one of MCMC trace plots and the other of
% current class assignments.  If trace_movie_file and progress_movie_file are provided 
% (as string file names) avi movies of the MCMC trace and the class
% assignments will be automatically generated and saved

% throughout the code there are chunks that are commented out -- this is
% to enhance efficiency in most cases.

% warning('Alpha updating is currently postponed until after then 10th iteration.')

if(nargin < 2)
    num_sweeps = 1000;
end

% set alpha gamma prior parameters (vague, following Navarro)
if(nargin < 3)
    a_0 = 10^-10;
end
if(nargin <4)
    b_0 = 10^-10;
end

% set normal inverse wishart hyper parameters
if(nargin < 5)
    mu_0 = zeros(size(training_data(:,1)));
end
if(nargin < 6)
    k_0 = 1;
end
if(nargin < 7)
    v_0 = size(training_data(:,1),1);
end

if v_0 < size(training_data(:,1),1)
    warning(['v_0 must be equal to or larger than the dimension'...
        'Warning: of the data\n v_0 changed from %d to %d'],v_0,...
        size(training_data(:,1),1));
    v_0 = size(training_data(:,1),1);
end

if nargin < 8
    lambda_0 = eye(size(training_data(:,1),1))*2;
end

% initialize
if nargin < 9
    alpha = 1;
else
    alpha = alpha_0;
end

if nargin < 10
    GRAPHICS = 0;
else
    GRAPHICS = 1;
end

if nargin < 12
    MOVIES = 0;
else
    MOVIES = 1;
end

if(GRAPHICS)
    figure(progress_plot_number)
    plot_mixture(training_data(1:2,:),ones(size(training_data(1,:))));

    cxlim = get(gca,'XLim');
    cylim = get(gca,'YLim');
end

% record movies?

% trace_movie_file = '/ltmp/trace.avi';
% progress_movie_file = '/ltmp/clusters.avi';

if(MOVIES)
    tmov = avifile(trace_movie_file, 'FPS', 5);
    cmov = avifile(progress_movie_file, 'FPS', 5);
    fig = figure(progress_plot_number);
    set(fig,'DoubleBuffer','on');
    set(gca,'xlim',cxlim,'ylim',cylim,...
        'NextPlot','replace','Visible','off');
    fig = figure(trace_plot_number);
    set(fig,'DoubleBuffer','on');
end




[D, N] = size(training_data);

% specify a memory efficient type for class_ids
class_id_type = 'uint16';
max_class_id = intmax(class_id_type);

% y = training_data;
% phi = cell(num_sweeps,1);
% D = size(training_data,1);
K_plus = 1;
class_id = zeros(N,num_sweeps,class_id_type);
K_record = zeros(num_sweeps,1);
alpha_record = zeros(num_sweeps,1);
lp_record = zeros(num_sweeps,1);

% seat the first customer at the first table
class_id(1,1) = 1;

% precompute student-T posterior predictive distribution constants
pc_max_ind = 1e5;
pc_gammaln_by_2 = 1:pc_max_ind;
pc_gammaln_by_2 = gammaln(pc_gammaln_by_2/2);
pc_log_pi = reallog(pi);
pc_log = reallog(1:pc_max_ind);




means = zeros(D,max_class_id);
sum_squares = zeros(D,D,max_class_id);
inv_cov = zeros(D,D,max_class_id);
log_det_cov = zeros(max_class_id,1);
counts = zeros(max_class_id,1,'uint32');
counts(1) = 1;
%
% lpY = lp_mvniw(class_id(:,1),training_data, mu_0, k_0, v_0, lambda_0);
% lpC = lp_crp(class_id(:,1),alpha);%-gamlike([a_0 b_0],alpha);
%
% lp_record(1) = lpY+lpC;
% alpha_record(1) = alpha;
% K_record(1) = length(unique(class_id(:,1)));

y = training_data(:,1);
yyT = y*y';
[lp ldc ic] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,1,y,yyT,k_0,mu_0,v_0,lambda_0);

means(:,1) = y;
sum_squares(:,:,1) = y * y';
counts(1) = 1;
log_det_cov(1) = ldc;
inv_cov(:,:,1) = ic;

yyT = zeros(D,D,N);
d2 = D/2;

% pre-compute the probability of each point under the prior alone
p_under_prior_alone = zeros(N,1);

Sigma = (lambda_0*(k_0+1)/(k_0*(v_0-2+1)))';
v = v_0-2+1;
mu = mu_0;
log_det_Sigma = log(det(Sigma));
inv_Sigma = inv(Sigma);
vd = v+D;

for i=1:N
    y = training_data(:,i);
    yyT(:,:,i) = y*y';

    if vd < pc_max_ind
        lp = pc_gammaln_by_2(vd) - (pc_gammaln_by_2(v) + d2*pc_log(v) + ...
            d2*pc_log_pi) - .5*log_det_Sigma-...
            (vd/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
    else
        lp = gammaln((v+d)/2)-(gammaln(v/2) + (d/2)*log(v) + ...
            (d/2)*pc_log_pi)-.5*log_det_Sigma-...
            ((v+d)/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
    end

    %     tll = gammaln((v+D)/2)-(gammaln(v/2) + (D/2)*log(v) + ...
    %         (D/2)*log(pi))-.5*log(det(Sigma))-...
    %         ((v+D)/2)*log(1+(1/v)*(y-mu)'*inv(Sigma)*(y-mu));

    %assert(lp==tll);

    p_under_prior_alone(i) = lp;

end

% initialize timers
time_1_obs = 0;
total_time = 0;


% run the Gibbs sampler
for sweep = 1:num_sweeps

    E_K_plus = mean(K_record(1:sweep));

    total_time = total_time + time_1_obs;
    if sweep==1
%         disp(['CRP Gibbs:: Sweep: ' num2str(sweep) '/' num2str(num_sweeps) ]);
    elseif mod(sweep,2000)==0
        rem_time = (time_1_obs*.05 + 0.95*(total_time/sweep))*num_sweeps-total_time;
        if rem_time < 0
            rem_time = 0;
        end
        disp(['CRP Gibbs:: Sweep: ' num2str(sweep) '/' num2str(num_sweeps) ', Rem. Time: '...
            secs2hmsstr(rem_time) ', Ave. Time: ' ...
            secs2hmsstr((total_time/(sweep))) ', Elaps. Time: ' ...
            secs2hmsstr(total_time) ', E[K^+] ' num2str(E_K_plus)]);
    end
    tic

    %     disp(['Gibbs Sampler Sweep ' num2str(sweep) '/' num2str(num_sweeps)])


    % for each datapoint, unseat it and reseat it, potentially generating a
    % new table
    si = 1;
    if sweep==1
        si =2;
    else
        class_id(:,sweep) = class_id(:,sweep-1);
    end

    for i=si:N
        y = training_data(:,i);
        % compute the CRP prior - start by computing the datapoints at
        % tables

        % count the number of points sitting on table k
        %         m_k = counts(1:K_plus);

        old_class_id= class_id(i,sweep);

        if old_class_id ~= 0
            counts(old_class_id) = counts(old_class_id) -1;

            if counts(old_class_id)==0
                % delete the table and compact all data structures
                % FWOOD work here
                hits = class_id(:,sweep)>=old_class_id;
                class_id(hits,sweep) = class_id(hits,sweep)-1;
                K_plus = K_plus-1;

                hits = [1:old_class_id-1 old_class_id+1:(K_plus+1)];
                means(:,1:K_plus) = means(:,hits);
                means(:,K_plus+1) = 0;
                sum_squares(:,:,1:K_plus) = sum_squares(:,:,hits);
                sum_squares(:,:,1+K_plus) = 0;
                counts(1:K_plus) = counts(hits);
                counts(K_plus+1) = 0;
                % DEAL WITH LDC and LDETCOV here aswell

                log_det_cov(1:K_plus) = log_det_cov(hits);
                log_det_cov(K_plus+1) = 0;
                inv_cov(:,:,1:K_plus) = inv_cov(:,:,hits);
                inv_cov(:,:,K_plus+1) = 0;


            else
                means(:,old_class_id) = (1/(double(counts(old_class_id))))...
                    *((double(counts(old_class_id))+1)*means(:,old_class_id) - y);
                sum_squares(:,:,old_class_id) = sum_squares(:,:,old_class_id) - yyT(:,:,i);
            end
        end

        % complete the CRP prior with new table prob.
        if sweep ~= 1
            prior = [double(counts(1:K_plus)); alpha]/(N-1+alpha);
        else
            prior = [double(counts(1:K_plus)); alpha]/(i-1+alpha);
        end

        likelihood = zeros(length(prior),1);
        %         temp = phi{sweep};

        % as per Radford's Alg. 3 we will compute the posterior predictive
        % probabilities in two scenerios, 1) we will evaluate the
        % likelihood of sitting at all of the existing tables by computing
        % the probability of the datapoint under the posterior predictive
        % distribution with all points sitting at that table considered and
        % 2) we will compute the likelihood of the point under the
        % posterior predictive distribution with no observations

        for ell = 1:K_plus
            % get the class ids of the points sitting at table l
            n = double(counts(ell));
            %             if n~=0
            m_Y = means(:,ell);
            mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*m_Y;
            k_n = k_0+n;
            v_n = v_0+n;

            % set up variables for Gelman's formulation of the Student T
            % distribution
            v = v_n-2+1;
            mu = mu_n;


            % if old_class_id == ell means that this point used to sit at
            % table ell, all of the sufficient statistics have been updated
            % in sum_squares, counts, and means but that means that we have
            % to recompute log_det_Sigma and inv_Sigma.  if we reseat the
            % particle at its old table then we can put the old
            % log_det_Sigma and inv_Sigma back, otherwise we need to update
            % both the old table and the new table
            if old_class_id ~= 0
                if old_class_id == ell
                    S = (sum_squares(:,:,ell) - n*m_Y*m_Y');
                    zm_Y = m_Y-mu_0;
                    lambda_n = lambda_0 + S  + ...
                        k_0*n/(k_0+n)*(zm_Y)*(zm_Y)';
                    Sigma = (lambda_n*(k_n+1)/(k_n*(v_n-2+1)))';

                    old_class_log_det_Sigma = log_det_cov(old_class_id);
                    old_class_inv_Sigma = inv_cov(:,:,old_class_id);

                    log_det_Sigma = log(det(Sigma));
                    inv_Sigma = inv(Sigma);
                    log_det_cov(old_class_id) = log_det_Sigma;
                    inv_cov(:,:,old_class_id) = inv_Sigma;
                else
                    log_det_Sigma = log_det_cov(ell);
                    inv_Sigma = inv_cov(:,:,ell);
                end
            else
                % this case is the first sweep through the data
                S = (sum_squares(:,:,ell) - n*m_Y*m_Y');
                zm_Y = m_Y-mu_0;
                lambda_n = lambda_0 + S  + ...
                    k_0*n/(k_0+n)*(zm_Y)*(zm_Y)';
                Sigma = (lambda_n*(k_n+1)/(k_n*(v_n-2+1)))';

                log_det_Sigma = log(det(Sigma));
                inv_Sigma = inv(Sigma);
                log_det_cov(ell) = log_det_Sigma;
                inv_cov(:,:,ell) = inv_Sigma;
            end

            vd = v+D;
            if vd < pc_max_ind
                lp = pc_gammaln_by_2(vd) - (pc_gammaln_by_2(v) + d2*pc_log(v) + ...
                    d2*pc_log_pi) - .5*log_det_Sigma-...
                    (vd/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
            else
                lp = gammaln((v+d)/2)-(gammaln(v/2) + (d/2)*log(v) + ...
                    (d/2)*pc_log_pi)-.5*log_det_Sigma-...
                    ((v+d)/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
            end

            likelihood(ell) = lp;
            
        end
        likelihood(K_plus+1) = p_under_prior_alone(i);

        likelihood = exp(likelihood-max(likelihood));
        likelihood = likelihood/sum(likelihood);

        % compute the posterior over seating assignment for datum i
        posterior = prior.*likelihood; % this is actually a proportionality
        % normalize the posterior
        posterior = posterior/sum(posterior);

        % pick the new table
        cdf = cumsum(posterior);
        rn = rand;

        new_class_id = find((cdf>rn)==1,1);

        if new_class_id > max_class_id
            error(['K^plus has exceeded the maximum value of ' class_id_type ])
        end

        counts(new_class_id) = counts(new_class_id)+1;

        means(:,new_class_id) = means(:,new_class_id)+ (1/double(counts(new_class_id)))*(y-means(:,new_class_id));
        sum_squares(:,:,new_class_id) = sum_squares(:,:,new_class_id) + yyT(:,:,i);

        if new_class_id == K_plus+1
            K_plus = K_plus+1;
        end

        if old_class_id == new_class_id
            % we don't need to compute anything new as the point was
            % already sitting at that table and the matrix inverse won't
            % change
            log_det_cov(old_class_id) = old_class_log_det_Sigma;
            inv_cov(:,:,old_class_id) = old_class_inv_Sigma;
        else
            % the point changed tables which means that the matrix inverse
            % sitting in the old_class_id slot is appropriate but that the
            % new table matrix inverse needs to be updated
            n = double(counts(new_class_id));
            %             if n~=0
            m_Y = means(:,new_class_id);
            k_n = k_0+n;
            v_n = v_0+n;

            % set up variables for Gelman's formulation of the Student T
            % distribution
            S = (sum_squares(:,:,new_class_id) - n*m_Y*m_Y');
            zm_Y = m_Y-mu_0;
            lambda_n = lambda_0 + S  + ...
                k_0*n/(k_0+n)*(zm_Y)*(zm_Y)';
            Sigma = (lambda_n*(k_n+1)/(k_n*(v_n-2+1)))';

            log_det_cov(new_class_id) = log(det(Sigma));
            inv_cov(:,:,new_class_id) = inv(Sigma);
        end

        % record the new table
        class_id(i,sweep) = new_class_id;
    end
    lZ = lp_mvniw(class_id(:,sweep),training_data, mu_0, k_0,v_0,lambda_0);
    lp = lp_crp(class_id(:,sweep),alpha);%-  gamlike([a_0 b_0],alpha);

    lp = lp+lZ;

    % MCMC Alpha update code 
    %         alpha_prop = alpha + randn*.001
    %
    %         if(alpha_prop < 0)
    %             lp_alpha_prop = -Inf;
    %         else
    %             lp_alpha_prop = lp_crp(class_id(:,sweep),alpha_prop);
    %             lp_alpha_prop = lp_alpha_prop -  gamlike([a_0 b_0],alpha_prop);
    %         end
    %         log_acceptance_ratio = lp_alpha_prop - lp;
    %         disp(['prop accepting new alpha = ' num2str(exp(log_acceptance_ratio))]);
    %         if(log(rand)<min(log_acceptance_ratio,0))
    %             alpha = alpha_prop;
    %         end

    nu = betarnd(alpha,N)+eps;
    % this is the same as eqn. 14 of Escobar and West 1994 Bayesian
    % Density Estimation and Inference Using Mixtures
   if sweep > 50
        alpha = 1/gamrnd(a_0+K_plus-1,b_0-log(nu));
   end

    % record the current parameters values
    K_record(sweep) = K_plus;
    alpha_record(sweep) = alpha;
    lp_record(sweep) = lp;

    if(GRAPHICS)
        figure(trace_plot_number)
        subplot(3,1,1)
        plot(1:sweep,lp_record(1:sweep));
        title('Log P')
        subplot(3,1,2)
        plot(1:sweep,K_record(1:sweep));
        title('K');
        subplot(3,1,3)
        plot(1:sweep,alpha_record(1:sweep));
        title('alpha');

        if(MOVIES)
            drawnow
            F = getframe(gcf);
            tmov = addframe(tmov,F);
        end


        figure(progress_plot_number)
        clf
        plot_mixture(training_data(1:2,:),class_id(:,sweep));
        set(gca,'XLim',cxlim);
        set(gca,'YLim',cylim);

        if(MOVIES)
            figure(progress_plot_number)
            drawnow

            F = getframe(gcf);
            cmov = addframe(cmov,F);

            figure(progress_plot_number)
            drawnow
            F = getframe(gcf);
            tmov = addframe(tmov,F);
        end

    end


    time_1_obs = toc;
    


    
    
end

if(MOVIES)
    close(tmov);
    close(cmov);
end
