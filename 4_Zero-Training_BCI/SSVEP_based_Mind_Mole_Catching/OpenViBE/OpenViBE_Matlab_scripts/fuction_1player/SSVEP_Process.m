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
%         interval = box_in.user_data.interval;  
        marker  = box_in.user_data.marker;


        %%%%%%%%%%%%%%   prep_segmentation set
%          varargin1= {'interval', interval};

         dat= box_in.inputs{1};
%           disp(box_in.inputs{1}.buffer{1}.matrix_data(1:4,1:10))

         dat_x = dat.buffer{1}.matrix_data;
%          disp('input')
%          disp(size(dat_x))
         dat_int = linspace(1000*dat.buffer{1}.start_time ,1000*dat.buffer{1}.end_time, size(dat_x,2));

         [box_in, ~, ~, ~] = OV_popInputBuffer(box_in,1);

        box_in.outputs{1}.header = box_in.inputs{1}.header;
        box_in.outputs{1}.header.nb_channels = 2;
        box_in.outputs{1}.header.channel_names = {'index_num','index_num2'};
        box_in.user_data.is_headerset = 1;

        SMT.x = dat_x ;
        SMT.ival = dat_int;

        for kkk=1:4
            subplot(4,2,2*kkk-1)
            plot(SMT.ival,SMT.x(kkk,:))
        end


        %
        FFT_fs = fs;
%         FFT_T=1/FFT_fs;
        FFT_L=size(SMT.x(1,:),2);

%         FFT_t=(0:FFT_L-1)*FFT_T;
%         FFT_fbp=[5 30];
        FFT_f=FFT_fs*(0:(FFT_L/2))/FFT_L;

        for kkk=1:4
        FFT_Y=fft(SMT.x(kkk,:));
        p2=abs(FFT_Y/FFT_L);
        p1=p2(:,1:FFT_L/2+1);
        p1(:,2:end-1)=2*p1(:,2:end-1);

        subplot(4,2,2*kkk)
        plot(FFT_f,p1(1,:))
        xlim([1 30])
        end
        box_in.user_data.m_cnt = box_in.user_data.m_cnt + 1;
%         fname = sprintf('sig%02d.fig', box_in.user_data.m_cnt);

%         saveas(gcf,fname);


%         numTrials = size(SMT.x, 2);
%         count= numTrials;

        in={'marker',marker;'freq', freq;'fs', fs;'time',time};

        varargin = in;
        if iscell(varargin) % cell to struct
            [nParam temp]=size(varargin);
            for i = 1:nParam
                str = varargin{i,1};
                opt.(str)= varargin{i,2};
            end
        end


        in =[opt];
        data = SMT.x;

        nClasses = size(in.marker,1);
        Freq = in.freq;

        t= 0: 1/in.fs :in.time;
        Y = cell(1,nClasses);
        r = cell(1,nClasses);

        for i = 1:nClasses
            ref = 2*pi*Freq(i)*t;
            Y{i} = [sin(ref);cos(ref);sin(ref*2);cos(ref*2)];
        end


        for j = 1:nClasses
            [~,~,r{j}] = canoncorr(data',Y{j}(:,1:fs*time)');
            r{j} = max(r{j});
        end

         cca_result  = cell2mat(r);
         [~, ind] = max(cca_result);

          box_in = OV_addOutputBuffer(box_in,1,dat.buffer{1}.start_time,dat.buffer{1}.start_time+1,ind*ones(2,fs*time));
          disp('output')
          disp(ind)
    end
          box_out = box_in;
      
end 
