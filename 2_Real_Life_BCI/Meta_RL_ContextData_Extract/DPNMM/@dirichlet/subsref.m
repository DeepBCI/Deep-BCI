function b = subsref(a,index)
%POISSON/SUBSREF Field name indexing for Dirichlet distribution objects

switch index.type
%     case '()'
%         b = a.ps(index.subs{:});
%     case '.'
%         if(strcmp(index.subs,'covariance'))
%             b = a.c;
%         elseif(strcmp(index.subs,'mean'))
%             b = a.m;
%         end
    otherwise
        error('Cell and name indexing not supported by Dirichlet distribution objects')
end
