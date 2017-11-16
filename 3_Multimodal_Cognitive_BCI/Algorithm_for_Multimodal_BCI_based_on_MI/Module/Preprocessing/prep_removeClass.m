function [ out ] = prep_removeClass( dat, class )
% prep_removeClass (Pre-processing procedure):
% 
% Description:
%     prep_removeClass removes specific classes
% 
% Examele:
%     dat = prep_removeClass(dat,{'right','foot'});
% 
% Input:
%     dat   - data structure
%     class - class names you want to remove
% 
% Output:
%     out - data structure, designated classes removed
% 
% Seon Min Kim, 11-2016
% seonmin5055@gmail.com

% if iscellstr(class)
%     warning('check the class parameter, it should be a string');
% end

if ndims(dat.x)==2 && length(dat.chan)>1
    type='cnt';
elseif ndims(dat.x)==3 || (ndims(dat.x)==2 && length(dat.chan)==1)
    type='smt';
end

if ischar(class)
    rm_Class{1}=class;
else
    rm_Class=class(:);
end

out=dat;
n_c=ones(1,length(dat.y_class));
for i=1:length(rm_Class)
    n_c= n_c .* ~ismember(dat.y_class, rm_Class{i});
end
n_c=logical(n_c);
%     msg=sprintf('OpenBMI: class "%s" is deleted',rm_Class{i});
%     disp(msg);
    
    if strcmp(type, 'smt')
        if isfield(dat, 'x')
            out.x=dat.x(:,n_c,:);
        end
    end    
    
    if isfield(dat, 't')
        out.t=dat.t(n_c);
    end
    if isfield(dat, 'y_dec')
        out.y_dec=dat.y_dec(n_c);
    end
    if isfield(dat, 'y_logic')
        tm=~ismember(dat.class(:,2),rm_Class);
        out.y_logic=dat.y_logic(tm,n_c);
    end
    if isfield(dat, 'y_class')
        out.y_class=dat.y_class(n_c);
    end
    if isfield(dat, 'class')
        tm=~ismember(dat.class(:,2),rm_Class);
        out.class=dat.class(tm,:);
    end


end


