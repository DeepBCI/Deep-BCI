function [lp, log_det_Sigma, inv_Sigma] = lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,n,m_Y,SS,k_0,mu_0,v_0,lambda_0, log_det_Sigma, inv_Sigma)
%
% function [lp, log_det_Sigma, inv_Sigma] =
% lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,n,m_Y,SS,k_0,mu_0,v_0,lambda_0, log_det_Sigma, inv_Sigma)
%
% Computes the posterior predictive distribution under the MVN-IW model
% given observations.  This function is tightly coupled with
% particle_filter.m and collapsed_gibbs_sampler.m and should not really be
% used in other contexts.  The variables are briefly described here:
%
%  pc_max_ind : maximum argument to precomputed functions passed in
%  pc_gammaln_by_2 : precomputed gammaln(arg)/2
%  pc_log_pi : precomputed log(pi)
%  pc_log : precomputed log(arg)
%  y : datapoint for which the posterior predictive score is wanted
%  n : number of datapoints already observed
%  m_Y : the arithmetic mean of the datapoints alread observed
%  SS : the sum of the squares of the datapoints (for computational
%  efficiency purposes
%  k_0,mu_0,v_0,lambda_0: the parameters to the MVNIW model following
%  Gelman pg 87
%  log_det_Sigma, inv_Sigma: if the function needs to be called multiple
%  times for the same set of data (previously observed datapoints) then the
%  output log_det_Sigma and inv_Sigma from the first calls can be passed in
%  as arguments in subsequent calls to save computation
%
% returns:
%   lp: the log posterior predictive probably of y under the model 
%    log_det_Sigma, inv_Sigma: partial computations used in computing this
%    value.  These are unique per m_Y, SS, k_0,mu_0,v_0, lambda_0 tuple

% persistent pc_gammaln_by_2 pc_log pc_log_pi pc_max_ind
% 
% % pre-compute a bunch of the gammaln stuff so that time isn't wasted
% % every call recomputing the same stuff
% if isempty(pc_gammaln_by_2)
%     mlock
%     pc_max_ind = 1e6;
%     pc_gammaln_by_2 = 1:pc_max_ind;
%     pc_gammaln_by_2 = gammaln(pc_gammaln_by_2/2);
%     pc_log_pi = reallog(pi);
%     pc_log = reallog(1:pc_max_ind);
% end

d = size(y,1);
if n~=0
    mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*m_Y;
    k_n = k_0+n;
    v_n = v_0+n;

    S = (SS - n*m_Y*m_Y');
    zm_Y = m_Y-mu_0;
    lambda_n = lambda_0 + S  + ...
        k_0*n/(k_0+n)*(zm_Y)*(zm_Y)';
else
    mu_n = mu_0;
    k_n = k_0;
    v_n = v_0;
    lambda_n = lambda_0;
end

% set up variables for Gelman's formulation of the Student T
% distribution
Sigma = lambda_n*(k_n+1)/(k_n*(v_n-2+1));
v = v_n-2+1;
mu = mu_n;

% if a precomputed det_Sigma and inv_Sigma haven't been passed in then
% compute them and also return the log probability of y under the student-T
% posterior predictive distribution
if nargin<13
    log_det_Sigma = reallog(det(Sigma));
    inv_Sigma = inv(Sigma);
% else
%     disp(['pre-computed: ' num2str(log_det_Sigma) ', check: ' num2str(log(det(Sigma)))])
end


% disp(['v: ' num2str(v) 'd: ' num2str(d)])
% compute the students T log likelihood y_i under the posterior
% predictive distribution given the prior parameters and all
% other datapoints currently sitting at that table

% if the values have been precomputed use them
vd = v+d;
if vd < pc_max_ind
    d2 = d/2;
    lp = pc_gammaln_by_2(vd) - (pc_gammaln_by_2(v) + d2*pc_log(v) + ...
        d2*pc_log_pi) - .5*log_det_Sigma-...
        (vd/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
else
    lp = gammaln((v+d)/2)-(gammaln(v/2) + (d/2)*log(v) + ...
        (d/2)*pc_log_pi)-.5*log_det_Sigma-...
        ((v+d)/2)*reallog(1+(1/v)*(y-mu)'*inv_Sigma*(y-mu));
end

