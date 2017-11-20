function lp = lp_crp(c,alpha)
% function lp = lp_crp(c,alpha)
%
%   takes a compact vector of class id's and the CRP parameter alpha
%   and returns the log probability of that partitioning
%
table_identifier = unique(c);
K_plus = length(table_identifier);
N = length(c);
m_k = zeros(K_plus,1);

for k = 1:K_plus
    m_k(k) = sum(c==table_identifier(k));
end

%the m_k ==0 case requires some care
foo = gammaln(m_k-1);
foo((m_k-1)==0)=0; % definition of log(0!)

lp = K_plus*log(alpha) + sum(foo) + gammaln(alpha) - gammaln(N+alpha);