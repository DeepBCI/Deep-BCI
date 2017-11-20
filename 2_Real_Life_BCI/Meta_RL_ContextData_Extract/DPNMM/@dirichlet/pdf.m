function [ret] = pdf(d,x)
% DIRICHLET/PDF 
%
tol = 1e-10;
if max(sum(x,2)) > 1+tol || min(sum(x,2)) < 1-tol
    error('Argument must sum to one.')
end


ret = (1/d.Z)*prod(x.^(repmat(d.alpha,size(x,1),1)-1),2);