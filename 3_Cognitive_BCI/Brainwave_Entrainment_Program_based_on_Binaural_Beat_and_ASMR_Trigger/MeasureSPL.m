%load and calibrate the data
[wav,fs_Hz]=audioread('ASMR_river_10min.mp3');  %load the WAV file
my_cal_factor = 1.0;  %the value for your system to convert the WAV into Pascals
wav_Pa = wav * my_cal_factor;
 
%extract the envelope
smooth_sec = 0.125;  %"FAST" SPL is 1/8th of second.  "SLOW" is 1 second;
smooth_Hz = 1/smooth_sec;
[b,a]=butter(1,smooth_Hz/(fs_Hz/2),'low');  %design a Low-pass filter
wav_env_Pa = sqrt(filter(b,a,wav_Pa.^2));  %rectify, by squaring, and low-pass filter
 
%compute SPL
Pa_ref = 20e-6;  %reference pressure for SPL in Air
SPL_dB = 10.0*log10( (wav_env_Pa ./ Pa_ref).^2 ); % 10*log10 because signal is squared
 
%plot results
figure;
subplot(2,1,1);
t_sec = ([1:size(wav_Pa)]-1)/fs_Hz;
plot(t_sec,wav_Pa);
xlabel('Time (sec)');
ylabel('Pressure (Pa)');
 
subplot(2,1,2)
plot(t_sec,SPL_dB);
xlabel('Time (sec)');
ylabel('SPL (dB)');
yl=ylim;ylim(yl(2)+[-80 0]);