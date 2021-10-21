function box_out = Process_2P(box_in)

for i = 1: OV_getNbPendingInputChunk(box_in,1)
    if(~box_in.user_data.is_headerset)
        box_in.outputs{1}.header = box_in.inputs{1}.header;
        box_in.outputs{1}.header.nb_channels = 5;
        % output - 2 classified results [2(players) x 1]
        box_in.outputs{1}.header.channel_names = {'P1'};
        box_in.user_data.is_headerset = 1;
    end
    
    % disp(box_in.inputs{1}.header)
    time = box_in.user_data.time;
    freq = box_in.user_data.freq;
    fs = box_in.inputs{1}.header.sampling_rate;    %CNT.EEG_SSVEP_test.fs;
    %         interval = box_in.user_data.interval;
    marker  = box_in.user_data.marker;
    
    %         disp(marker);
    
    
    %%%%%%%%%%%%%%   prep_segmentation set
    %          varargin1= {'interval', interval};
    
    dat= box_in.inputs{1};
    
    dat_x = box_in.inputs{1}.buffer{1}.matrix_data;
    
    for subband_idx = 1: 4
        dat_1p(:,:,subband_idx) = dat_x(1+10*(subband_idx-1):  5 +10*(subband_idx-1) , :) ;
        dat_2p(:,:,subband_idx) = dat_x(6+10*(subband_idx-1): 10 +10*(subband_idx-1) , :) ;
        
    end
    
%     for i = 1: 4
%         dat_ch(:,:,i) =       % ch X time X sub-band
%     end
    
    dat_int = linspace(1000*dat.buffer{1}.start_time ,1000*dat.buffer{1}.end_time, size(dat_x,2));
    %          pop 4 input
    for i = 1
        [box_in, start_time,end_time , ~] = OV_popInputBuffer(box_in,i);
    end

    % disp('buffer');
    temp_mat =[];
    for i = 1:size(box_in.inputs{2}.buffer,2)
        if (size(box_in.inputs{2}.buffer{i}.matrix_data,2) > 1)
            disp((box_in.inputs{2}.buffer{i}.matrix_data));
            temp_marker=(box_in.inputs{2}.buffer{i}.matrix_data);
            %                     disp(size(temp_marker))
            %                     temp_mat = cat(2,temp_mat,temp_marker(1)) ;
        end
    end
    %          disp()
    
    %          disp(temp_size);
    
    %           disp(box_in.inputs{1}.buffer{1}.matrix_data(1:4,1:10))
    
    dat_x = dat.buffer{1}.matrix_data;
    %          disp('input')
    %          disp(size(dat_x))
    dat_int = linspace(1000*dat.buffer{1}.start_time ,1000*dat.buffer{1}.end_time, size(dat_x,2));
    %         disp('first')
    %         disp(box_in.inputs{2})
    [box_in, ~, ~, ~] = OV_popInputBuffer(box_in,1);
    %         disp('second')
    %         disp(box_in.inputs{2})
    temp_len = size(box_in.inputs{2}.buffer,2);
    for i =1 : temp_len
        [box_in, ~, ~, ~] = OV_popInputBuffer(box_in,2);
    end
    %         disp('third')
    %         disp(size(box_in.inputs{2}.buffer))
    
    box_in.outputs{1}.header = box_in.inputs{1}.header;
    box_in.outputs{1}.header.nb_channels = 1;
    box_in.outputs{1}.header.channel_names = {'CCA_index_num'};
    box_in.user_data.is_headerset = 1;
    
    SMT.x = dat_x ;
    SMT.ival = dat_int;

    box_in.user_data.m_cnt = box_in.user_data.m_cnt + 1;

    
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
    
    nClasses = size(in.freq,2);
    Freq = in.freq;
    
    
    
    t= 0: 1/fs :time;
    Y = cell(1,nClasses);
    r = cell(1,nClasses);
    
    
    if (size(dat_1p,2)< length(t))
        t = t(1:end - abs(length(t) - size(dat_1p,2)));
    elseif (size(dat_1p,2)> length(t))
        dat_1p = dat_1p(:, 1:end -abs(length(t) - size(dat_1p,2)),:);
        dat_2p = dat_2p(:, 1:end -abs(length(t) - size(dat_2p,2)),:);
    end
    
    
    
    used_marker = [];
    player_marker = [];
    for i =1 : size(temp_marker,1)
        if temp_marker(i,1) >=10
            used_marker = cat(2, used_marker,temp_marker(i,1)-10);
        end
        if temp_marker(i,1) < 10 && temp_marker(i,1) > 1
            player_marker = temp_marker(i,1) -1 ;
        end   
    end
    disp('player')
    if player_marker == 1 
        dat_ch = dat_1p;
        disp(player_marker)
    else
        dat_ch = dat_2p;
        disp(player_marker)
    end
    
    
    %disp(used_marker)
    %         temp_marker(:,1) -10; % used freq, ex 1,2,3,4 +10
    %disp(length(used_marker))
    
    for i = 1:nClasses
        ref = 2*pi*Freq(i)*t;
        Y{i} = [sin(ref);cos(ref);sin(ref*2);cos(ref*2) ;sin(ref*3);cos(ref*3);sin(ref*4);cos(ref*4) ];
    end
    
    %         for i =1 : 36
    %             disp(ismember(i,used_marker))
    %         end
    
    a= 1;b=0; % weight parameter
    for sub_idx =1 :4
        wn(sub_idx) = sub_idx^(-a)+b;
    end
%     disp('dat size')
%     disp(size(dat_ch))
%     disp(size(Y{1}))
    disp('size- used_marker')
    disp(length(used_marker))
    disp(unique(used_marker))
    
    if (length(used_marker)>=1 )
        for class_idx = 1:nClasses
            if (ismember(class_idx,used_marker))
                for subband_idx =1 : 4
                    [~,~,r] = canoncorr(dat_ch(:,:,subband_idx)',Y{class_idx}');
                    results(subband_idx)=max(r);
                end
            else
                results=[0 0 0 0];
            end
            weight_r(class_idx)= sum((wn.*results).^2);
        end
        [~,ind]=max(weight_r);
%         [~, ind] = max(cca_result);
    else
        weight_r=[];
        ind = 0;
    end

    disp('results')
    disp(ind)
    
    box_in = OV_addOutputBuffer(box_in,1,dat.buffer{1}.start_time,dat.buffer{1}.end_time,ind*ones(1,size(data,2)));
    
    
    
    
end
box_out = box_in;

end
