function d = dirichlet(l,k)
%Dirichlet distribution class constructor.
%   d = DIRICHLET(alpha,k) creates a dirichlet distribution with a vector of 
%   parameters alpha if alpha is a vector or a vector of weights equal to alpha/k
%   (with length k) if alpha is a scalar

% Author: Frank Wood fwood@gatsby.ucl.ac.uk

switch nargin
    case 1
        if(isa(l,'dirichlet'))
            d.alpha = l.alpha;
        else

            d.alpha = l;
        end

    case 2
        d.alpha = ones(1,k)*l/k;
    otherwise
        error('Incorrect arguments to Dirichlet constructor');
end
            if sum(d.alpha<0) > 0
                error('Dirichlet parameters must be positive')
            end
bc = distribution();
d.Z = prod(gamma(d.alpha))/gamma(sum(d.alpha));
d.logZ = sum(gammaln(d.alpha)) - gammaln(sum(d.alpha));
d = class(d,'dirichlet',bc);

