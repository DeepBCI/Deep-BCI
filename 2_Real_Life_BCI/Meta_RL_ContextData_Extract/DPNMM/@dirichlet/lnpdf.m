function [ret] = lnpdf(d,x)
% DIRICHLET/LNPDF 
%
tol = 1e-20;
if sum(x) > 1+tol 
    error('Argument must sum to one.')
end


ret = -d.logZ+sum(log(x).*(d.alpha-1));