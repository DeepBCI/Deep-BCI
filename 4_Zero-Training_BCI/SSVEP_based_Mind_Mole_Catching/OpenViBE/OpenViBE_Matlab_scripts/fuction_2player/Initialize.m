function box_out = Initialize(box_in)
% Initialization

    box_in.user_data.time = 4;
    box_in.user_data.freq = [10, 12, 15, 7];
    box_in.user_data.marker  = {'1','up';'2', 'left';'3', 'right';'4', 'down'};
    box_in.user_data.interval = [0 4000];  % ms
%     box_in.user_data.fs = 300; %
%     box_in.user_data.fs = box_in.inputs{1}.header.sampling_rate;
    box_in.user_data.m_cnt = 0;
    box_in.user_data.is_headerset = false;
	
	% We also add some statistics
	box_in.user_data.nb_matrix_processed = 0;
	box_in.user_data.mean_fft_matrix = 0; 
    
    box_out = box_in;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
