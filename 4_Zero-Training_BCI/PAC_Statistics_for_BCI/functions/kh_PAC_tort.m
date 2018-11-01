function [MI] = kh_PAC_tort(Phase, Amp, nbins)
% Caculate phasse-sorted amplitude plot
% and get KL distance from the uniform distribution
%
% Input:
%    - Phase: phase time series extracted from Hilbert
%    - Amp: amplitude envelope time series extracted from Hilbert
%    - nbins: number of phase bins (Tort recommended 18 bins)
% Output:
%    - modulation index (MI)

position = zeros(1, nbins);
winsize = 2* pi / nbins;

for i=1:nbins
    position(i) = -pi + (i-1) * winsize;
end
[MI, ~] = ModIndex_v2(Phase, Amp, position);

end