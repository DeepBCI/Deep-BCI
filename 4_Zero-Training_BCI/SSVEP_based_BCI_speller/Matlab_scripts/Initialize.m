function box_out = Initialize(box_in)
% Initialization
    data_length = 3.0;
    box_in.user_data.time = data_length;
    box_in.user_data.freq =  [14 15 16 17 18 19 20 21 22];
    box_in.user_data.marker  = {'1','up';'2', 'left';'3', 'right';'4', 'down'};
    box_in.user_data.interval = [0 data_length*1000];  % ms
%     box_in.user_data.fs = 300; %box_in.inputs{1}.header.sampling_rate;
    box_in.user_data.m_cnt = 0;


%     for i=1:size(box_in.settings,2)
% %        fprintf('\t%s : %s\n',box_in.settings(i).name, num2str(box_in.settings(i).value));
%           disp('\t%s : %s\n',box_in.settings(i).name, num2str(box_in.settings(i).value)')
%     end
    %  %%%%% eeglab 
%      cd('C:/Users/Bio_lab_HG/Documents/MATLAB/fieldtrip-20190618'); % FieldTrip
%      ft_defaults;    
%      cd('C:/Users/Bio_lab_HG/Documents/MATLAB/eeglab2019_0'); % EEGLAB
%      close all; % close EEGLAB GUI
     
     disp('Loading DSI chanlocs..');
%     
    box_in.user_data.is_headerset = false;
	
	% We also add some statistics
	box_in.user_data.nb_matrix_processed = 0;
	box_in.user_data.mean_fft_matrix = 0; 
    
    box_out = box_in;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
