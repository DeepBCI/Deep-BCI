function [datapoints labels means covariances] = sample_igmm_prior(n,a_0,b_0,mu_0,lambda_0,k_0,v_0)
%
% function [datapoints labels means covariances] = sample_igmm_prior(n,a_0,b_0,mu_0,lambda_0,k_0,v_0)
%
%  This function draws n samples from the infinite Gaussian mixture model parameterized by
%
%    alpha ~ Gamma(a_0, b_0)
%    labels ~ CRP(alpha)
%    class_i_covariance ~ Inverse Wishart (lambda_0, v_0)
%    class_i_mean ~ Normal(m_0, class_i_covariance /  k_0)
%    datapoints(label == i) ~ Normal(class_i_mean, class_i_covariance)
%
%  The dimensionality of the data is implicit in the mu_0 and lambda_0
%  arguments (which are not themselves checked for consistency)
%
%   In addition to the class labels and the datapoints, the means (KxD) and
%   the covariances (KxDxD) are returned where K is the number of classes
%   and D is the dimensionality of the datapoint
% 
%
alpha = gamrnd(a_0,b_0);
labels = sample_crp(n,alpha);

dim = length(mu_0);
num_tables = length(unique(labels));

means = zeros(num_tables,dim);
covariances = zeros(num_tables,dim,dim);
datapoints = zeros(n,dim);

for t = 1:num_tables
    covariances(t,:,:) =iwishrnd(lambda_0,v_0);
    means(t,:) = mvnrnd(mu_0,squeeze(covariances(t,:,:))/k_0);
    num_points_at_table_t = sum(labels == t);
    datapoints(labels == t,:) = mvnrnd(means(t,:),squeeze(covariances(t,:,:)),num_points_at_table_t);

end
