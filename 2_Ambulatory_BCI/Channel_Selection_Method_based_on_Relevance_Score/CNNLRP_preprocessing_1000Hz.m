
% 1. load data (length 518 based on visual cue ) & adjust target index ( 1,2 -> 0,1)
% 2. select 18channles ( 18 of 118) , apply butterworth filtering
% 3. apply padded_stft then images makes 200 x 200 by stft parameters
% 4. save the images shpae = [280, 18, 200, 200]

clear all
load('data_set_IVa_av');
load('true_labels_av');

% index 1,2 -> 0,1 조정
ttrue_y=true_y-1;

% mrk 구성 
% pos, y, class_name{right, foot}
mrk.y = ttrue_y;

cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;

% cue 1x280 -> 280x1
cue=transpose(cue);
temp=[];
cnt=cnt;
numt=280;

% 각 visual cue 위치에서 5180개씩 사용(5s ,1000hz )
% 각 visual cue의 시작포인트가 550이 최소간격. 즉 하나의 time segment에 하나의 visual cue만 포함
for k=1:numt
    temp=cnt(cue(k):cue(k)+5180,:);
    %transpose
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end


st=1;
stt=1;
% ll :left, rr :right 나누는것인듯

for k=1:numt
    if mrk.y(k)==1
        ll(st)=k;
        st=st+1;
    else
        rr(stt)=k;
        stt=stt+1;
    end
end

l=length(ll);
r=length(rr);

% right / foot 나눠서 각각 eeg1/eeg2에 넣는과정
for k=1:l
    ee1(:,:,k)=eeg(:,:,ll(k));
end

for k=1:r
    ee2(:,:,k)=eeg(:,:,rr(k));
end

% 처음에 118부터 비교를해야 fair 하다고 말씀하셔서 잠시 채널지움
%chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93];


% R: window length
R=1200;
window=hamming(R);

% hop size (overlapping =100points 임)
hop = 80;

% fft resolution
N = 1024;


% 여기서 채널잘라냄
% for k=1:18
%     e11(k,:,:)=ee1(chn(k),:,:);
%     e22(k,:,:)=ee2(chn(k),:,:);
% end


% second order butterworth
% devided the signal into frames (buffer function)
[bbb,aaa]=butter(4,[7/500 30/500]);
[ddd,ccc]=butter(4,[140/500 160/500]);

% padded_stft -> 양쪽40이 아니라 뒤쪽으로 80넣음 (window size :120)
for channel=1:118
    
    for left_trial_idx=1:l
        % e11.shape = [18, 501, 140] = [ n_channel, time_seq, trial]
        % bandpass filtering
        % e1.shape = [left_trial_idx, 501]
        % 채널 118개 사용시 아래 명령어사용
        mu_beta_e1(left_trial_idx,:)=filtfilt(bbb,aaa,ee1(channel, :, left_trial_idx));  
        gamma_e1(left_trial_idx,:)=filtfilt(ddd,ccc,ee1(channel, :, left_trial_idx));
        
        e1(left_trial_idx,:) = gamma_e1(left_trial_idx,:) + mu_beta_e1(left_trial_idx,:);
        
        %e1(left_trial_idx,:)=ee1(channel, :, left_trial_idx);  
     
        
        % Custom STFT 
        % singal에 windowing 해서 segments 만들고 segments에 zeropadding 해서 fft
        [S1, F1, T1] = padded_stft( e1(left_trial_idx,:), window, hop, N, 1000 );
        spectro_left_data(left_trial_idx, channel, :,:) = zscore(abs(S1(1:200,:)).^2);
        
%         temp_S1(:,:) = normalize(abs(S1(1:200,:)).^2, 'scale');
        
%         for f_idx = 1:200
%             for t_idx = 1:50
%                 if temp_S1(f_idx, t_idx) < 0.5
%                     temp_S1(f_idx,t_idx) = 0;
%                 end
%             end
%             
%         end
%         
%         spectro_left_data(left_trial_idx, channel, :,:) = temp_S1;
%         
%        
        
    end
    
    for right_trial_idx=1:r
        % bandpass filtering
        % 채널 118개 사용시 아래 명령어사용
        mu_beta_e2(right_trial_idx,:)=filtfilt(bbb,aaa,ee2(channel, :, right_trial_idx));  
        gamma_e2(right_trial_idx,:)=filtfilt(ddd,ccc,ee2(channel, :, right_trial_idx));  
        
        e2(right_trial_idx,:) = gamma_e2(right_trial_idx,:) + mu_beta_e2(right_trial_idx,:);
        %e2(right_trial_idx,:)=ee2(channel, :, right_trial_idx);  
    
        % Custom STFT
        [S2, F2, T2] = padded_stft( e2(right_trial_idx,:), window, hop, N, 1000);
        spectro_right_data(right_trial_idx, channel, :, : ) = zscore(abs(S2(1:200,:)).^2);
        
%         temp_S2(:,:) = normalize(abs(S2(1:200,:)).^2, 'scale');
%         
%         for f2_idx = 1:200
%             for t2_idx = 1:50
%                 if temp_S2(f2_idx, t2_idx) < 0.5
%                     temp_S2(f2_idx,t2_idx) = 0;
%                 end
%             end
%             
%         end        
%         spectro_right_data(right_trial_idx, channel, :, : ) = temp_S2;
%         
    
    end
        
end


spectro_train(1:l,:,:,:) = spectro_left_data;
spectro_train(l+1 : l+r , : ,:, : ) = spectro_right_data;
save('STFT_av_train.mat','spectro_train', '-v7.3');

spectro_label=[ones(l,1);ones(r,1)+1]-1;
save('STFT_av_label.mat','spectro_label', '-v7.3');


% imagesc y_axis reversed
% set(gca,'YDir','normal') 
