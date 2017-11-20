function c = find_optimal_c(Q,N)

Q = sort(Q,'descend');
c = 0;
k = 0;
M = length(Q);

k_old = -inf;

while k_old ~=k
    k_old = k;
    c = (N-k)/sum(Q(k+1:end));
    k = k+ sum(Q(k+1:M)*c > 1);
end

