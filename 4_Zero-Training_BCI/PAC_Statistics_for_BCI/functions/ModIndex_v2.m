% Programmed by Adriano Tort, CBD, BU, 2008
% 
% Phase-amplitude cross-frequency coupling measure:
%
% [MI,MeanAmp]=ModIndex_v2(Phase, Amp, position)
%
% Inputs:
% Phase = phase time series
% Amp = amplitude time series
% position = phase bins (left boundary)
%
% Outputs:
% MI = modulation index (see Tort et al PNAS 2008, 2009 and J Neurophysiol 2010)
% MeanAmp = amplitude distribution over phase bins (non-normalized)
 
function [MI,MeanAmp]=ModIndex_v2(Phase, Amp, position)

nbin=length(position);  % we are breaking 0-360o in 18 bins, ie, each bin has 20o
winsize = 2*pi/nbin;
 
% now we compute the mean amplitude in each phase:
 
MeanAmp=zeros(1,nbin); 
for j=1:nbin   
I = find(Phase <  position(j)+winsize & Phase >=  position(j));
MeanAmp(j)=mean(Amp(I)); 
end
 
% so note that the center of each bin (for plotting purposes) is
% position+winsize/2
 
% at this point you might want to plot the result to see if there's any
% amplitude modulation
 
% bar(10:20:720,[MeanAmp,MeanAmp])
% xlim([0 720])

% and next you quantify the amount of amp modulation by means of a
% normalized entropy index:

MI=(log(nbin)-(-sum((MeanAmp/sum(MeanAmp)).*log((MeanAmp/sum(MeanAmp))))))/log(nbin);

end
