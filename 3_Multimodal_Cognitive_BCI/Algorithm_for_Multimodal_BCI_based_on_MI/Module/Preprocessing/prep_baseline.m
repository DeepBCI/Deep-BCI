function [out] = prep_baseline(dat,varargin)
% prep_baseline (Pre-processing procedure):
% 
% Description:
%     This function corrects the baseline by subtracting average amplitude
%     in the specified interval from a segmented signal.
%
% Example:
% [out] = prep_baseline(dat,{'Time',[-100 0];'Criterion','class'})
%
% Input:
%     dat       - segmented data structure
% Option:
%     Time      - time interval. [start ms, end ms] or time(ms) from the
%                 beginning (default: all)
%     Criterion - 'class', 'trial', 'channel' (default: 'trial')
%
% Returns:
%     dat - baseline corrected data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat,'x')
    warning('OpenBMI: Data must have fields named ''x''');return
elseif ndims(dat.x)~=3 && size(dat.chan,2)~=1
    warning('OpenBMI: Data must be segmented');
elseif ~isfield(dat,'ival')
    warning('OpenBMI: Data must have fields named ''ival''');return
elseif ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''fs''');return
end

opt = opt_cellToStruct(varargin{:});
if ~isfield(opt,'Time')
    opt.Time = [dat.ival(1),dat.ival(end)];
end
if ~isfield(opt,'Criterion')
    opt.Criterion = 'trial';
end
if isscalar(opt.Time)
    opt.Time = [dat.ival(1),dat.ival(1)+opt.Time];
elseif ~isvector(opt.Time)
    warning('OpenBMI: Time should be a scalar or a vector');return
end

[nT,~,~] = size(dat.x);
t = opt.Time-dat.ival(1)+1;
if t(1)<1 || ceil(t(end)*dat.fs/1000)>dat.ival(end)
    warning('OpenBMI: Selected time interval is out of time range');return
end

switch opt.Criterion
    case 'trial'
        idx = floor(t(1)*dat.fs/1000+1):ceil(t(end)*dat.fs/1000);
        base = nanmean(dat.x(idx,:,:),1);
        x = dat.x-repmat(base,[nT,1,1]);
    case 'class'
        if ~isfield(dat,'y_logic')
            warning('OpenBMI: Data must have fields named ''y_logic''');return
        end
        x = zeros(size(dat.x));
        for i=1:size(dat.y_logic,1)
            t_idx = floor(t(1)*dat.fs/1000)+1:ceil(t(end)*dat.fs/1000);
            k = find(dat.y_logic(i,:)==1);
            n = size(k,2);
            base = nanmean(nanmean(dat.x(t_idx,k,:),1),2);
            x(:,k,:) = dat.x(:,k,:)-repmat(base,[nT,n,1]);
        end
%     case 'channel'
%         base = 
end

out = rmfield(dat,'x');
out.x = x;
