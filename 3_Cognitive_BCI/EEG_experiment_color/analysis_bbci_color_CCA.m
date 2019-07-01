%% classify_try1이랑 같은 코드 (군더더기 없는 버전)

clear all; close all; clc; 

dd = 'C:\Users\HANSEUL\Desktop\DoYeun\2019\analysis\';
filelist = 'dy_color';

%% Preprocessing cnt 
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); % Load cnt, mrk, mnt variables to Matlab

% Parameter setting 
filtBank = [0.5 40];  % band pass filtering할 주파수 대역 설정
ival = [2800 5000]; % sampling rate이 1000 이므로 마커 기준 2초를 잘라야 하니까 0~2000

% 넣고 싶은 채널 넣음
subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
%  subChannel = [57, 58, 25, 61, 62, 63, 29, 30, 31]; % 비주얼 콜텍스

[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);
%[cnt, mrk] =proc_resample(cnt, 100);


% IIR Filter (Band-pass filtering)
cnt = proc_filtButter(cnt, 5, filtBank);


% Channel Selection
cnt = proc_selectChannels(cnt, subChannel);   

% MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set
mnt = mnt_adaptMontage(mnt, cnt);


%% cnt to epoch    
epo = cntToEpo(cnt, mrk, ival);

% % 베이스라인 안하고 regularization할 경우 (위에 -200을 0으로 바꾸고 오세요)
% ival22 = [0 2000];
% epo = proc_baseline(epo, ival22); 

% -200ms를 기준으로 baseline correction
 base = [2800 3000];
 epo = proc_baseline(epo, base);
 
 ival2 = [3000 5000]; 
 epo = proc_selectIval(epo,ival2);

%% 클래스 고르기

%epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Toilet');

% 전체 클래스
%  epo_all = proc_selectClasses(epo, 'white','red', 'green', 'blue', 'yellow', 'cyan', 'magenta');
epo_all = proc_selectClasses(epo, 'red', 'green', 'blue');
% 6클래스씩
 %epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Clock', 'imagine_Light', 'imagine_Toilet', 'imagine_TV', 'imagine_Water');
 %epo_all = proc_selectClasses(epo, 'imagine_Hello', 'imagine_Helpme', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Yes');


%% 각 클래스별 88트라이얼씩 골라내기

% 먼저, 각 클래스 별 몇 번의 트라이얼이 찍혔는지 세어보자
count_epo=sum(epo_all.y,2);

% 88개 이상을 다 잘라버리자
y_temp = zeros(size(epo_all.y));

for ci=1:size(y_temp,1)
    cidx = find(epo_all.y(ci,:)==1);
    cidx = cidx(1:50);
    y_temp(:,cidx) = epo_all.y(:,cidx);
end

idx = find(sum(y_temp(:,:),1)==1);
y = y_temp(:,idx);
x = epo_all.x(:,:,idx);

epo_all.x = x;
epo_all.y = y;

count_epo=sum(epo_all.y,2);


%% reference signals for SSVEP 
window_time=2;

fs = epo.fs;
freq = [2 3 4 6 12];

% t=[0:window_time/(window_time*fs-1):window_time];
t = [0:1/fs:window_time];

% ground truth
Y=cell(1);
freq = [2 3 4 6 12]; % 60Hz를 5로 나눈 Frequency-> 12Hz, 7로 나눈 Frequency -> 8.57
t = [0:1/fs:window_time]; % 1/fs 부터 window_time까지 1/fs의 Frequency로 sampling
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end

%% y_dec
nTrial = size(epo_all.y,2);
epo_all.y_dec=ones(1,nTrial);
for i=1:nTrial
    epo_all.y_dec(i) = find(epo_all.y(:,i) == 1);
end
%% accuracy
% ear
% initialization
r_corr = []; r=[]; r_value=[]; pred=[];
nTrial=size(epo_all.x,3);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo_all.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    
    [r_value(i), pred(i)]=max(r(i,:));
end
acc_all=length(find(epo_all.y_dec == pred))/nTrial*100