function a = subsasgn(a,index,val)
% POISSON/SUBSASGN Define index assignment for Dirichlet distribution objects

switch index.type
case '()'
%     ps = a.ps;
%     ps(index.subs{:}) = val;
%     if(sum(ps)~=1)
%         warning('New multinomial does not sum to 1')
%     end
%     a.ps(index.subs{:}) = val;
    otherwise
        error('Cell and name indexing not supported by Dirichlet distribution objects')
        
end
