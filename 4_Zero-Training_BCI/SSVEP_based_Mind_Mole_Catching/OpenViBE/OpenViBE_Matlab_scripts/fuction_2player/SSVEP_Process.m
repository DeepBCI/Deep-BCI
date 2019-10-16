function box_out = Process(box_in)

for i = 1: OV_getNbPendingInputChunk(box_in,1)
    if(~box_in.user_data.is_headerset)
        box_in.outputs{1}.header = box_in.inputs{1}.header;
        box_in.outputs{1}.header.nb_channels = 2;
        % output - 2 classified results [2(players) x 1]
        box_in.outputs{1}.header.channel_names = {'P1','P2'};
        box_in.user_data.is_headerset = 1;
    end
    
    % disp(box_in.inputs{1}.header)
    time = box_in.user_data.time;
    freq = box_in.user_data.freq;
    fs = box_in.inputs{1}.header.sampling_rate;    %CNT.EEG_SSVEP_test.fs;
%     interval = box_in.user_data.interval;
%     marker  = box_in.user_data.marker;
    labels_set = {'P3', 'P4','O1', 'O2'};
    %%%%%%%%%%%%%%   prep_segmentation set
%     varargin1= {'interval', interval};
    
    [box_in, start_time, end_time, matrix_data] = OV_popInputBuffer(box_in, 1);
    dat_x = matrix_data;
%     dat_int = linspace(1000*start_time ,1000*end_time, size(dat_x,2));
    
    % Data = [8 x 1200]
    % P1 = [1:4 x 1200] / P2 = [5:8 x 1200]
    player_ind = [1:4; 5:8];
    input_dat = dat_x'; % [time x ch]
%     SMT.ival = dat_int;
    
    %---------------------------------- display FFT amplitude for P1 and P2
    FFT_fs = fs;
    Player_FFT = cell(0);
    %     disp(size(input_dat));
    %     disp(size(matrix_data));
    for nbuser = 1:2
        FFT_L = size(input_dat, 1);
        FFT_Y = fft(input_dat(:, player_ind(nbuser,:)));
        FFT_f=FFT_fs*(0:(FFT_L/2))/FFT_L;
        
        temp = abs(FFT_Y/FFT_L);
        P1 = temp(1:FFT_L/2+1,:);
        P1(2:end-1,:) = 2*P1(2:end-1,:); % single-sided FFT
        Player_FFT{nbuser} = P1;
        
        subplot(1,2,nbuser); plot(FFT_f, Player_FFT{1}); xlim([0 40]);
        ylabel('FFT amplitude'); xlabel('Frequency (Hz)');
        legend(labels_set);
        box off;
        set(gca, 'fontsize', 13);
        title(sprintf('Player %d FFT amplitude', nbuser));
    end
    %     Player_FFT{1} = P1 FFT amplitude / Player_FFT{2} = P2 FFT amplitude
    %
    %     fname = sprintf('sig%02d.fig', box_in.user_data.m_cnt);
    %     saveas(gcf,fname);
    
    %------------------------------------------------- CCA (classification)
    nClasses = length(freq);
    Freq = freq;
    
    t= 0:1/fs:time;
    Y = cell(1,nClasses);
    r = cell(1,nClasses);
    for k = 1:nClasses
        ref = 2*pi*Freq(k)*t(1:end-1);
        Y{k} = [sin(ref); cos(ref); sin(ref*2); cos(ref*2)];
    end
    
    % canoncorr = observation (sample points) x ch
    user_answer = zeros(2,1);
    for nbuser = 1:2
        for j=1:nClasses
            [~,~,r{j}] = canoncorr(input_dat(:, player_ind(nbuser,:)),Y{j}(:,:)');
            r{j} = max(r{j});
        end
        cca_result = cell2mat(r);
        [~, ind] = max(cca_result);
        user_answer(nbuser) = ind;
    end

%     %     disp(size(ind));
    user_answer(1) = randi([1 3]);
    user_answer(2) = randi([1 3]);
%     user_answer(1) = 2;
%     user_answer(2) = 2;
    output = repmat(user_answer, 1, size(matrix_data, 2));
   
    box_in = OV_addOutputBuffer(box_in,1,start_time,end_time,output);
%     disp(user_answer);
end
box_out = box_in;
end
