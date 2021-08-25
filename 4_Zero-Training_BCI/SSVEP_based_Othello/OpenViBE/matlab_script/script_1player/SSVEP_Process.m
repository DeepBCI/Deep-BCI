function box_out = SSVEP_Process(box_in)	
%     tic;
    for i = 1: OV_getNbPendingInputChunk(box_in,1)
        if(~box_in.user_data.is_headerset)
            box_in.outputs{1}.header = box_in.inputs{1}.header;
            box_in.outputs{1}.header.nb_channels = 1;
            % output - 2 classified results [2(players) x 1]
            box_in.outputs{1}.header.channel_names = {'P1'};
            box_in.user_data.is_headerset = 1;
        end

    % disp(box_in.inputs{1}.header)
        time = box_in.user_data.time;
        freq = box_in.user_data.freq;
        fs = box_in.inputs{1}.header.sampling_rate;    %CNT.EEG_SSVEP_test.fs;
%         interval = box_in.user_data.interval;  
%         marker  = box_in.user_data.marker;


        
        
        %%%%%%%%%%%%%%   prep_segmentation set
%          varargin1= {'interval', interval};

         dat_1= box_in.inputs{1};  % first harmonic
%          dat_2= box_in.inputs{2};  % second harmonic
%          dat_3= box_in.inputs{3};  % third harmonic
%          dat_4= box_in.inputs{4};  % fourth harmonic

%          disp(size(dat_2.buffer{1}.matrix_data));
%         for i = 1:4
%             disp(size(box_in.inputs{i}.buffer{1}.matrix_data))
%         end
% %           disp(size(box_in.inputs{1}.buffer{1}.matrix_data))
%           disp(size(box_in.inputs{1}.buffer{1}.matrix_data))
          dat_x = box_in.inputs{1}.buffer{1}.matrix_data;
          for i = 1: 4
              dat_ch(:,:,i) = dat_x(1+5*(i-1): size(dat_x,1)/4 + 5*(i-1) , :) ;      % ch X time X sub-band
          end
         
         dat_int = linspace(1000*dat_1.buffer{1}.start_time ,1000*dat_1.buffer{1}.end_time, size(dat_x,2));
%          pop 4 input
         for i = 1 
             [box_in, start_time,end_time , ~] = OV_popInputBuffer(box_in,i);
         end
%          
%          
        box_in.outputs{1}.header = box_in.inputs{1}.header;
        box_in.outputs{1}.header.nb_channels = 1;
        box_in.outputs{1}.header.channel_names = {'index_num'};
        box_in.user_data.is_headerset = 1;
        
        SMT.x = dat_ch ;
        SMT.ival = dat_int;
% 
%         for kkk=1:4
%             subplot(4,2,2*kkk-1)
%             plot(SMT.ival,SMT.x(kkk,:))
%         end
% 
% 
%         %
%         FFT_fs = fs;
% %         FFT_T=1/FFT_fs;
%         FFT_L=size(SMT.x(1,:),2);
% 
% %         FFT_t=(0:FFT_L-1)*FFT_T;
% %         FFT_fbp=[5 30];
%         FFT_f=FFT_fs*(0:(FFT_L/2))/FFT_L;
% 
%         for kkk=1:4
%         FFT_Y=fft(SMT.x(kkk,:));
%         p2=abs(FFT_Y/FFT_L);
%         p1=p2(:,1:FFT_L/2+1);
%         p1(:,2:end-1)=2*p1(:,2:end-1);
% 
%         subplot(4,2,2*kkk)
%         plot(FFT_f,p1(1,:))
%         xlim([1 30])
%         end

        box_in.user_data.m_cnt = box_in.user_data.m_cnt + 1;
%         fname = sprintf('sig%02d.fig', box_in.user_data.m_cnt);
% 
% %         saveas(gcf,fname);
% 
% 
% %         numTrials = size(SMT.x, 2);
% %         count= numTrials;
% 
%         in={'marker',marker;'freq', freq;'fs', fs;'time',time};
% 
%         varargin = in;
%         if iscell(varargin) % cell to struct
%             [nParam temp]=size(varargin);
%             for i = 1:nParam
%                 str = varargin{i,1};
%                 opt.(str)= varargin{i,2};
%             end
%         end
% 
% 
%         in =[opt];
%         data = SMT.x;
        Freq = freq;
        nClasses = length(Freq);
        
% 
        t= 0: 1/ fs :time;
        t = t(1:end-1);
        Y = cell(1,nClasses);
        r = cell(1,nClasses);
% 
        for i = 1:nClasses
            ref = 2*pi*Freq(i)*t;
            Y{i} = [sin(ref);cos(ref);sin(ref*2);cos(ref*2) ;sin(ref*3);cos(ref*3);sin(ref*4);cos(ref*4)   ];
        end
%         
%         disp(size(dat_ch))
%         disp(size(Y{1}))
        a= 1;b=0;
        for class_idx = 1 : nClasses
            for subband_idx = 1: 4
            wn(subband_idx) = subband_idx^(-a)+b;
            
                [~,~,r]=canoncorr(dat_ch(:,:,subband_idx)', Y{class_idx}');
                results(subband_idx)=max(r); 
            end
            weight_r(class_idx)= sum((wn.*results).^2);
        end
        
        [~,detect_f]=max(weight_r);
        
%         disp(detect_f)

% 
%          cca_result  = cell2mat(r);
%          [~, ind] = max(cca_result);
          
%             disp(dat_1.buffer{1}.start_time)
%             disp(dat_1.buffer{1}.start_time + fs*time -1)
%             disp(detect_f*ones(1,fs*time))

          box_in = OV_addOutputBuffer(box_in,1,start_time, end_time, detect_f*ones(1,size(dat_ch,2) ));
          disp('output')
          disp(detect_f)
%           disp(toc)

    end
          box_out = box_in;
      
end 
