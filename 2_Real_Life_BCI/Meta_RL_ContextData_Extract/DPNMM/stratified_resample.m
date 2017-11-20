function [rx] = stratified_resample(x,w,N)
% function [rx] = stratified_resample(x,w,N)
%
%  returns N samples from the weighted particle set 
%  x,w with x (DxM) being M, D-dimensional samples and w being a vector of
%  M real weights that sum to one
%

[D, M] = size(x);
ni = randperm(M);
x = x(:,ni);
w = w(ni);
rx = zeros(D,N);
rw = zeros(1,N);
cdf = cumsum(w);
cdf(end) = 1;
p = linspace(rand*(1/N),1,N);
% p = sort(p);

picked = zeros(1,M);
j=1;
for i=1:N
    while j<M && cdf(j)<p(i)
        j=j+1;
    end
    picked(j) = picked(j)+1;
end

rind=1;
for i=1:M
    if(picked(i)>0)
        for j=1:picked(i)
            rx(:,rind) = x(:,i);
            rw(rind) = w(i);
            rind=rind+1;
        end
    end
end

rw = rw./sum(rw);


