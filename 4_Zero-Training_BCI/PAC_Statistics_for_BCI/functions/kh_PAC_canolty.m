function [MI] = kh_PAC_canolty(Phase, Amp)
% Apply MVL (mean vector length) from Canolty et al., 2006
%
% Input:
%    - Phase: phase time series extracted from Hilbert
%    - Amp: amplitude envelope time series extracted from Hilbert
% Output:
%    - modulation index (MI)

Z = Amp .* exp(1i * Phase); % get analytic signal
MI = abs(mean(Z));

end