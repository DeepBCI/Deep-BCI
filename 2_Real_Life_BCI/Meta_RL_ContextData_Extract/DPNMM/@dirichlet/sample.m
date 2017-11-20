function samples = sample(d,cases)
% DIRICHLET/SAMPLE 
%   samples = p(dist,cases) returns cases number of samples
%   from the Dirichlet dist.
if nargin ==1
    cases = 1;
end
num = gamrnd(repmat(d.alpha,cases,1),1);
samples = num./repmat(sum(num,2),1,length(d.alpha));