function val = get(s,propName)
% DIRICHLET/GET Get property from the specified  Dirichlet
% and return the value. Property names are: alpha, and "parameter vector"

switch propName
    case 'parameter vector'
        val = s.alpha;
            case 'parameter vector upper bounds'
        val = ones(size(s.alpha))*Inf;
            case 'parameter vector lower bounds'
        val = zeros(size(s.alpha));
    case 'alpha'
        val = s.alpha;
    otherwise
        error([propName ,'Is not a valid Dirichlet distribution property'])
end
