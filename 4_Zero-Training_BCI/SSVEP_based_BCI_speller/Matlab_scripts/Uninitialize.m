%
function box_out = Uninitialize(box_in)
    disp('Uninitializing the box...')
		
% 	Fs = box_in.inputs{1}.header.sampling_rate;        % Sampling frequency
% 	L = box_in.inputs{1}.header.nb_samples_per_buffer; % Length of signal
% 	
% 	NFFT = 2^nextpow2(L); 
% 	f = Fs/2*linspace(0,1,NFFT/2+1);
% 	
% 	box_in.user_data.mean_fft_matrix = box_in.user_data.mean_fft_matrix / box_in.user_data.nb_matrix_processed;
% 	
	%% we close the previous figure window and plot the mean FFT between 5 and 50Hz.
% 	close(gcf);
% 	plot_range_fmin = box_in.settings(3).value;
% 	plot_range_fmax = box_in.settings(4).value;
% 	plot(f(plot_range_fmin*2:plot_range_fmax*2),2*abs(box_in.user_data.mean_fft_matrix(plot_range_fmin*2:plot_range_fmax*2))) 
% 	title('MEAN Single-Sided Amplitude Spectrum of the corrupted signal (channel 1)')
% 	xlabel('Frequency (Hz)')
% 	ylabel('Amplitude')
	
	% We pause the execution for 10 seconds (to be able to see the figure before the scenario is stopped)
% 	pause(10);
	
    box_out = box_in;
end