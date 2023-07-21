clear all; close all; clc;

dd = 'data_destination_IMAGINEDspeech';
filelist = 'filename_IMAGINEDspeech';

dd2 = 'data_destination_VISUALimagery';
filelist2 = 'filename_VISUALimagery';

% Load cnt, mrk, mnt variables to Matlab
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); 
[cnt2, mrk2, mnt2]=eegfile_loadMatlab([dd2 filelist2]);

% Parameter setting 
filtBank = [0.5 40];
ival = [-200 2000];

% subchannel setting for the channel selection
 subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
%subChannel = [4 38 8 43 9 47 13 48 14 18 52 19 56 24 57];

% Downsampling
[cnt, mrk] =proc_resample(cnt, 100, 'mrk',mrk,'N',0);
[cnt2, mrk2] =proc_resample(cnt2, 100, 'mrk',mrk2,'N',0);

% Band-pass filtering
 cnt = proc_filtButter(cnt, 5, filtBank); 
 cnt2 = proc_filtButter(cnt2, 5, filtBank);
  
% Channel Selection
cnt = proc_selectChannels(cnt, subChannel);
cnt2 = proc_selectChannels(cnt2, subChannel);

% MNT_ADAPTMONTAGE
mnt = mnt_adaptMontage(mnt, cnt);
mnt2 = mnt_adaptMontage(mnt2, cnt2);

%% cnt to epoch    
epo = cntToEpo(cnt, mrk, ival);
epo2 = cntToEpo(cnt2, mrk2, ival);

%% baseline correction
base = [-200 -1];
epo = proc_baseline(epo, base); 
epo2 = proc_baseline(epo2, base);

ival2 = [0 2000];
epo = proc_selectIval(epo,ival2);
epo2 = proc_selectIval(epo2,ival2);

%% grouped by each class
epo_all = proc_selectClasses(epo, 'imagine_Ambulance','imagine_Clock','imagine_Hello', 'imagine_Helpme', 'imagine_Light', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Toilet', 'imagine_TV', 'imagine_Water', 'imagine_Yes', 'imagine_Rest');
epo_all2 = proc_selectClasses(epo2, 'imagine_Ambulance','imagine_Clock','imagine_Hello', 'imagine_Helpme', 'imagine_Light', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Toilet', 'imagine_TV', 'imagine_Water', 'imagine_Yes', 'imagine_Rest');
% epo_all = proc_selectClasses(epo, 'imagine_Ambulance', 'imagine_Light', 'imagine_Thankyou',  'imagine_Rest');
% epo_all2 = proc_selectClasses(epo, 'imagine_Ambulance', 'imagine_Light', 'imagine_Thankyou',  'imagine_Rest');

count_epo=sum(epo_all.y,2);    
count_epo2=sum(epo_all2.y,2);  

y_temp = zeros(size(epo_all.y));
for ci=1:size(y_temp,1)
    cidx_1 = find(epo_all.y(ci,:)==1);
    cidx_1 = cidx_1(1:88);
    y_temp(:,cidx_1) = epo_all.y(:,cidx_1);
end

idx = find(sum(y_temp(:,:),1)==1);
y = y_temp(:,idx);
x = epo_all.x(:,:,idx);

epo_all.x = x;
epo_all.y = y;

count_epo=sum(epo_all.y,2);     

y_temp2 = zeros(size(epo_all2.y));
for ci2=1:size(y_temp2,1)
    cidx2 = find(epo_all2.y(ci2,:)==1);
    cidx2 = cidx2(1:88);
    y_temp2(:,cidx2) = epo_all2.y(:,cidx2);
end
idx2 = find(sum(y_temp2(:,:),1)==1);
y2 = y_temp2(:,idx2);
x2 = epo_all2.x(:,:,idx2);
epo_all2.x = x2;
epo_all2.y = y2;
count_epo2=sum(epo_all2.y,2);  
%% Feature extraction - cross-validation
% [csp_fv, csp_w, csp_eig] = proc_multicsp(epo_all, 3); % basic multi-CSP
% [csp_fv2, csp_w2, csp_eig2] = proc_multicsp(epo_all2, 3); % basic multi-CSP
%%  CORRELATION
% extracting the envelope of the segmented epochs
% epo_all=proc_envelope(epo_all);
% epo_all2=proc_envelope(epo_all2);
    ci=1; 
    cii=2;
    idx_1 = find(epo_all.y(ci,:)==1);
    idx_2 = find(epo_all2.y(ci,:)==1);
    idx_3 = find(epo_all3.y(ci,:)==1);    
    iidx_1 = find(epo_all.y(cii,:)==1);
    iidx_2 = find(epo_all2.y(cii,:)==1);
    iidx_3 = find(epo_all3.y(cii,:)==1);        
    %ch = 25; %Pz
    ch = 13; %C3
    %ch = 14; %Cz
    %ch = 14; %Cz    
    ch13 = epo_all.x(:,ch,idx_1);
    ch13_2 = epo_all2.x(:,ch,idx_2);    
    cih13 = epo_all.x(:,ch,iidx_1);
    cih13_2 = epo_all2.x(:,ch,iidx_2);   
    ch13_grand = epo_all.x(:,ch,:);
    ch13_2_grand = epo_all2.x(:,ch,:);   
    m1 = mean(ch13,3);
    m2 = mean(ch13_2,3);    
    mi1 = mean(cih13,3);
    mi2 = mean(cih13_2,3);    
    mg1 = mean(ch13_grand,3);
    mg2 = mean(ch13_2_grand,3);
    [corr12, pval2] = corr(m1,m2)
    [corri12, pvial2] = corr(mi1,mi2)
     
%% Display
%     plot(m1, 'blue')
%     hold on;%     
%     plot(mi1, 'blue')
%     hold on;% 
%     plot(m2, 'black')
%     hold on;%     
%     plot(mi2, 'black')
%     hold on;%     
%     close all;
    
