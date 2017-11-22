function [out] = prep_selectTime(dat, varargin)
% prep_selectTime (Pre-processing procedure):
%
% Description:
% This function selects the part of a specific time interval
% from continuous or epoched data.
% (i)  For continuous data, this function selects data in specifie time
%      interval from the whole data.
% (ii) For epoched data, this function selects time interval in each trial.
%      If you want to select trials in specific time interval, you can use
%      a function 'prep_selectTrials'
%
% Example:
% out = prep_selectTime(dat, {'Time',[1000 3000]})
%
% Input:
%     dat - Data structure
%     time - Time interval to be selected (ms)
%
% Returns:
%     out - Time selected data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if isempty(varargin)
    warning('OpenBMI: Time interval should be specified')
    out = dat;
    return
end
opt = opt_cellToStruct(varargin{:});
if ~isfield(opt,'Time')
    warning('OpenBMI: Time interval should be specified.')
    return
end
ival = opt.Time;

if ~isfield(dat,'x') || ~isfield(dat,'t') || ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''x'',''t'',''fs''')
    return
end

if isfield(dat,'y_dec') && isfield(dat,'y_logic') && isfield(dat,'y_class')
    a=1;
else
    a=0;
    warning('OpenBMI: Data should have fields named ''y_dec'',''y_logic'',''y_class''')
end

d = ndims(dat.x);
is = ceil(ival(1)*dat.fs/1000);
ie = floor(ival(2)*dat.fs/1000);

if d == 3 || (d==2 && length(dat.chan)==1)
    if ival(1)<dat.ival(1) || ival(2)>dat.ival(end)
        warning('OpenBMI: Selected time interval is out of epoched interval')
        return
    end
    iv = [is:ie]-dat.ival(1)*dat.fs/1000+1;
    x = dat.x(iv,:,:);
    t = dat.t;
    time = iv/dat.fs*1000;
    if a
        y_dec = dat.y_dec;
        y_logic = dat.y_logic;
        y_class = dat.y_class;
    end
elseif d == 2 && length(dat.chan)>1
    if ival(1)<0 || ival(2)/1000>size(dat.x,1)/dat.fs
        warning('OpenBMI: Selected time interval is out of time range')
        return
    end
    x = dat.x(is:ie,:);
    s = find((dat.t*1000/dat.fs)>=ival(1));
    e = find((dat.t*1000/dat.fs)<=ival(2));
    iv = s(1):e(end);
    t = dat.t(iv);
    if a
        y_dec = dat.y_dec;
        y_logic = dat.y_logic;
        y_class = dat.y_class;
    end
end
out = rmfield(dat,{'x','t'});
out.x = x;
out.t = t;
if isfield(out,'ival')
    out.ival = time;
end
if a
    out.y_dec = y_dec;
    out.y_logic = y_logic;
    out.y_class = y_class;
end
