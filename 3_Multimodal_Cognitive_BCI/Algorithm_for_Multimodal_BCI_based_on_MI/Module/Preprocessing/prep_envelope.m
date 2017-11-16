function [out] = prep_envelope(dat,varargin)
% prep_envelope (Pre-processing procedure):
% 
% Description:
%     This function smoothly outlines the extremes of an oscillating
%     signal, continuous or epoched.
%
% Example:
% [out] = prep_envelope(dat)
%
% Input:
%     dat    - Oscillating signal
% Options:
%     Time[ms] - time window. scalar or nx1 vector for weighting (default: 100)
%     Method - 'centered' or 'causal' (default: causal)
%
% Returns:
%     out - Envelope of the signal
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end

s = size(dat.x);
dat.x= reshape(abs(hilbert(dat.x(:,:))),s);
out= prep_movingAverage(dat,varargin{:});

