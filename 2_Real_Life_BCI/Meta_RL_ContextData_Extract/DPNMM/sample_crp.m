function seating = sample_crp(n,alpha)
%
% function seating = sample_crp(n,alpha)
%
%  Returns the "seating" or class id's for n draws from a Chinese
%  restuarant process with parameter alpha
%

m_k = zeros(1,n);
m_k(1) = 1;
seating = zeros(1,n);
seating(1) = 1;

for i = 2:n
    p_table = [m_k(m_k~= 0) alpha];
    p_table = p_table/sum(p_table);
    seating(i) = find(cumsum(p_table) > rand,1,'first');
    m_k(seating(i)) = m_k(seating(i))+1;
end