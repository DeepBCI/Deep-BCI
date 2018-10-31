function [MI] = kh_PAC_PLV(Phase, Amp)
% Apply PLV (phase locking value) from Cohen et al., 2008
%
% Input:
%    - Phase: phase time series extracted from Hilbert
%    - Amp: amplitude envelope time series extracted from Hilbert
% Output:
%    - modulation index (MI)
%
% It extracts additional phase time series from amplitude envelope 
% and calculates PLV

amp_phase = angle(hilbert(detrend(Amp)));
MI = abs(mean(exp(1i*(Phase - amp_phase))));

end