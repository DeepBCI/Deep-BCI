function [ out ] = prep_selectClass( dat, varargin )
% prep_selectClass (Pre-processing procedure):
% 
% This function selects data of specified classes 
% from continuous or epoched data.
% 
% Example:
%     out = prep_selectClass(dat,{'class',{'right', 'left','foot'}});
% 
% Input: 
%     dat - Structure. Data which classes are to be selected
% Option
%     class - Name of classes that you want to select (e.g. {'right','left'})
% 

if isempty(varargin)
    error('OpenBMI: Classes should be specified');
end
opt=opt_cellToStruct(varargin{:});
if ~isfield(dat, 'class')
    error('OpenBMI: Data structure must have a field named ''class''');
end
if ~isfield(opt, 'class')
    error('OpenBMI: Classes should be specified in a correct form');
end
if ischar(opt.class) && strcmp(opt.class,'all')
    out=dat; sprintf('you chose all classes');return
elseif iscell(opt.class) && numel(opt.class)==1 && strcmp(opt.class{1},'all')
    out=dat; sprintf('you chose all classes');return
end



% % smkim
% cls_idx = find(ismember(dat.class(:,2),opt.class));
% tr_idx = find(ismember(dat.y_dec,cls_idx));
% out = prep_selectTrials(dat,{'Index',tr_idx});
% out.y_logic(~ismember(dat.class(:,2),opt.class),:) = [];



% if ndims(dat.x)==2
%     type='cnt';
% elseif ndims(dat.x)==3
%     type='smt';
% end

[n_c nn]=size(dat.class);
n_c=zeros(1, n_c);

%find class index
if ischar(opt.class) % one class
    temp=ismember(dat.class(:,2),opt.class);
    [a b]=find(temp==1);
    n_c(a)=1;
    clear a b;
else
    for i=1:length(opt.class)
        temp=ismember(dat.class(:,2),opt.class{i});
        [a b]=find(temp==1);
        n_c(a)=1;
        clear a b;
    end
end

[n_d]=find(n_c==0);
del_classes=cell(length(n_d),1);

for i=1:length(n_d)
    del_classes{i}=dat.class{n_d(i),2};
end


out=prep_removeClass(dat,del_classes);


end