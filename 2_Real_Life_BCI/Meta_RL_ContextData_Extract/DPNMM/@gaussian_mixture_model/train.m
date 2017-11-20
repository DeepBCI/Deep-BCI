function [gmm,cluster_assignments, log_likelihood] = train(gmmin,data,num_mixture_components,max_em_iterations,use_spectral_clustering)
% GAUSSIAN_MIXTURE_MODEL/TRAIN
%    gmm = TRAIN(gmmin,data,num_mixture_components,max_em_iterations)

% Copyright October, 2006, Brown University, Providence, RI. 
% All Rights Reserved 

% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a commercial
% product is hereby granted without fee, provided that the above copyright
% notice appear in all copies and that both that copyright notice and this
% permission notice appear in supporting documentation, and that the name of
% Brown University not be used in advertising or publicity pertaining to
% distribution of the software without specific, written prior permission. 

% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE. IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY
% SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
% RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
% CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
% CONNECTION WITH THE USE.

% Author: Frank Wood fwood@cs.brown.edu

if(nargin<4)
    max_em_iterations = 100;
end
if(nargin<3)
    error('Need at least num_mixture_components, data, and gmmin');
end
if(nargin<5)
    use_spectral_clustering=1;
end
if(use_spectral_clustering)
    [cluster_assignments, log_likelihood, mixture_means, mixture_covariances, alpha] = cluster_em_gaussian_mixture(data,num_mixture_components,[],0,max_em_iterations);
else
    n = size(data,1);
    k = num_mixture_components;
    initial_cluster_inds = ceil(rand(n,1)*k);
    initial_clusters = ones(n,k)*.01;
    for(i=1:n)
        initial_clusters(i,initial_cluster_inds(i)) = .95;
    end
    [cluster_assignments, log_likelihood, mixture_means, mixture_covariances, alpha] = cluster_em_gaussian_mixture(data,num_mixture_components,initial_clusters,0,max_em_iterations);
end

evalstr = 'gmm = gaussian_mixture_model(';

for(i=1:num_mixture_components)
    if(i<num_mixture_components)
        evalstr = [evalstr num2str(alpha(i)) ',gaussian(' mat2str(mixture_means(i,:)) ''',' mat2str(mixture_covariances{i}) '''), '];
    else
        evalstr = [evalstr num2str(alpha(i)) ',gaussian(' mat2str(mixture_means(i,:)) ''',' mat2str(mixture_covariances{i}) ''')'];
    end
end
evalstr = [evalstr ');'];
eval(evalstr);
