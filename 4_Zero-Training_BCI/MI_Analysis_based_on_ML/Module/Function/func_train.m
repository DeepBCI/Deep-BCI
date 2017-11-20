function [ out ] = func_train( fv, varargin )
% func_train:
%     Training the features of data by classifier.In this version, only producing
%     the lda classifier. Other classifier algorithm will be updated.
% 
% Example:
%     func_train(fv, {'classifier','lda'};
%
% Input:
%     fv- training features for making classifier
%
% Options:
%     classifier - setting the classifier.
%
% Retuns:
%     out - parameter of trained classifier.

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:}
end

switch lower(opt.classifier)
    case 'lda' %only binary class
%         out.cf_param=train_RLDAshrink(fv.x,fv.mrk.logical_y);
% BBCI toolbox
out.cf_param=func_RLDA(fv.x,fv.y_logic);
        out.classifier='LDA';
end

end


