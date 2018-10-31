function y = get_raw_simulation_PAC(len, srate, PM_freq, AM_freq, noise)
% get_simulation_PAC(): returns sinosoidal simulation data for PAC
%  input:
%         len: data length
%         srate: sammplig rate
%         PM_freq: phase modulating frequency (LF)
%         AM_freq: amplitude modulating frequency (HF)
%         noise: noise variance of Gaussian noise (mu=0)
%             default: 0
%  output:
%         y: simulation data

if nargin < 5
    noise = 0;
end
nonmodulatedamplitude=2;
% increase this to get less modulation; 
% you'll see that this is reflected in the MI value
t = 1:1:len;

lfp = (0.2*(sin(2*pi*t*PM_freq/srate)+1)+nonmodulatedamplitude*0.1)...
    .*sin(2*pi*t*AM_freq/srate)+sin(2*pi*t*PM_freq/srate);
additive_noise = noise * randn(size(lfp)) + 0;
lfp = lfp + additive_noise;
y = lfp;
end