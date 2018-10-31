function [MI] = kh_PAC_ozkurt(Phase, Amp)
% Apply MVL (mean vector length) and normalizes by amplitude power
% from Ozkurt et al., 2011
%
% This method is the same as Canolty's idea, but adds normalization
% tum (amplitude power) to reduce the impact of amplitude
%
% Input:
%    - Phase: phase time series extracted from Hilbert
%    - Amp: amplitude envelope time series extracted from Hilbert
% Output:
%    - modulation index (MI)

N = length(Amp);
Z = Amp .* exp(1i * Phase); % get analytic signal
MI = (1/sqrt(N)) * abs(mean(Z)) / sqrt(mean(Amp.*Amp));

end