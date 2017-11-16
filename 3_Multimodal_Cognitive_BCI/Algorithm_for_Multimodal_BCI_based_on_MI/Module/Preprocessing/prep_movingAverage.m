function [out] = prep_movingAverage(dat,varargin)
% prep_movingAverage (Pre-processing procedure):
% 
% Description:
% This function averages the data with moving time window
% 
% Example:
% [out] = prep_movingAverage(dat,{'Time',80;'Method','centered'})
% 
% Input:
%     dat - data structure, continuous or epoched
% Options:
%     Time[ms] - time window. scalar or nx1 vector for weighting (default: 100)
%     Method - 'centered' or 'causal' (default: causal)
%     Samples - the number of samples in a single time window. scalar
%               If the 'Time' option exists, this option will be ignored.
% 
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

% 1. window에 weighting하는 경우 고려 필요. window는 scalar나 vector
% 2. samples 수 고려. time이랑 같이 들어오면 무시
% 3. NaN이 포함될 경우.

if isempty(varargin)
    opt.Time = 100;
    opt.Method = 'causal';
else
    opt = opt_cellToStruct(varargin{:});
end
if ~isfield(opt,'Time')
    opt.Time = 100;
end
if ~isfield(opt,'Method')
    opt.Method = 'causal';
end

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if ~isfield(dat,'fs')
    warning('OpenBMI: Data structure must have a field named ''fs''')
    return
end

if ~ismatrix(dat.x)==2 && ~ndims(dat.x)==3
    warning('OpenBMI: Data dimension must be 2 or 3');return
end

if ndims(dat.x)==3
    xx=zeros(size(dat.x));
    for i=1:size(dat.x,2)
        x=squeeze(dat.x(:,i,:));
        temp=rmfield(dat,'x');
        temp.x=x;
        temp2=prep_movingAverage(temp,varargin{:});
        xx(:,i,:)=temp2.x;
    end
    out = rmfield(dat,'x');
    out.x = xx;
    return
end

[t,~] = size(dat.x);
n = round(opt.Time*dat.fs/1000);
x = zeros(size(dat.x));

switch opt.Method
    case 'causal'
        for i=1:min(n,t)
            x(i,:) = mean(dat.x([1:i],:),1);
        end
        for i=n+1:t
            x(i,:) = mean(dat.x([i-n+1:i],:),1);
        end
    case 'centered'
        ws = -floor(n/2);
        we = ws+n-1;
        for i=1:-ws+1
            x(i,:) = mean(dat.x([1:i+we],:),1);
        end
        for i=-ws+2:t-we
            x(i,:) = mean(dat.x([i+ws:i+we],:),1);
        end
        for i=t-we+1:t
            x(i,:) = mean(dat.x([i+ws,end],:),1);
        end
end

out = rmfield(dat,'x');
out.x = x;
