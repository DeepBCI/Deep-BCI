function s = set(s,varargin)
% Dirichlet/SET Set Dirichlet distribution properties to the specified values
% and return the updated object


propertyArgIn = varargin;
while length(propertyArgIn) >= 2,
    prop = propertyArgIn{1};
    val = propertyArgIn{2};
    propertyArgIn = propertyArgIn(3:end);
    switch prop
        case 'alpha'
            s.alpha = val;
            d.Z = prod(gamma(d.alpha))/gamma(sum(d.alpha));
            d.logZ = sum(lngamma(d.alpha)) - lngamma(sum(d.alpha));
        case 'parameter vector'
            s.alpha = val;
            d.Z = prod(gamma(d.alpha))/gamma(sum(d.alpha));
            d.logZ = sum(lngamma(d.alpha)) - lngamma(sum(d.alpha));
        otherwise
            error('Invalid Dirichlet property')
    end
end
