%% Plot the tsne plot of the CSP feature vectors

clear all; close all; clc;

% load the data to be analyzed
dd = 'data_destination_IMAGINEDspeech';
filelist = 'filename_IMAGINEDspeech';

dd2 = 'data_destination_VISUALimagery';
filelist2 = 'filename_VISUALimagery';

dd3 = 'data_destination_OVERTspeech';
filelist3 = 'filename_OVERTspeech';

% Load cnt, mrk, mnt variables to Matlab
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist]); 
[cnt2, mrk2, mnt2]=eegfile_loadMatlab([dd2 filelist2]);
[cnt3, mrk3, mnt3]=eegfile_loadMatlab([dd3 filelist3]);

% Parameter setting 
filtBank = [0.5 40];  % band pass filtering
%filtBank = [70 500];  % band pass filtering (High frequency)
ival = [-200 2000]; % epoch setting considering the sampling rate

% subchannel setting for the channel selection
 subChannel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ...
           24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64];
%subChannel = [4 38 8 43 9 47 13 48 14 18 52 19 56 24 57]; % this channel selection showed the highest correlation in my data (broca's and wernicke's area)

% Downsampling
[cnt, mrk] =proc_resample(cnt, 256, 'mrk',mrk,'N',0);
[cnt2, mrk2] =proc_resample(cnt2, 256, 'mrk',mrk2,'N',0);
[cnt3, mrk3] =proc_resample(cnt3, 256, 'mrk',mrk3,'N',0);

% Band-pass filtering (IIR Filter)
 cnt = proc_filtButter(cnt, 5, filtBank); 
 cnt2 = proc_filtButter(cnt2, 5, filtBank);
 cnt3 = proc_filtButter(cnt3, 5, filtBank);
  
% Channel Selection
cnt = proc_selectChannels(cnt, subChannel);
cnt2 = proc_selectChannels(cnt2, subChannel);
cnt3 = proc_selectChannels(cnt3, subChannel);

% MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set
mnt = mnt_adaptMontage(mnt, cnt);
mnt2 = mnt_adaptMontage(mnt2, cnt2);
mnt3 = mnt_adaptMontage(mnt3, cnt3);

%% cnt to epoch    
epo = cntToEpo(cnt, mrk, ival);
epo2 = cntToEpo(cnt2, mrk2, ival);
epo3 = cntToEpo(cnt3, mrk3, ival);

%% baseline correction
base = [-200 0];
epo = proc_baseline(epo, base); 
epo2 = proc_baseline(epo2, base);
epo3 = proc_baseline(epo3, base);

ival2 = [0 2000];
epo = proc_selectIval(epo,ival2);
epo2 = proc_selectIval(epo2,ival2);
epo3 = proc_selectIval(epo3,ival2);

%% grouped by each class

epo_all = proc_selectClasses(epo, 'imagine_Ambulance', 'imagine_Light', 'imagine_Thankyou',  'imagine_Rest');
epo_all2 = proc_selectClasses(epo, 'imagine_Ambulance', 'imagine_Light', 'imagine_Thankyou',  'imagine_Rest');
epo_all3 = proc_selectClasses(epo, 'imagine_Ambulance', 'imagine_Light', 'imagine_Thankyou',  'imagine_Rest');

% counting the number of trials in each classes
count_epo=sum(epo_all.y,2);    
count_epo2=sum(epo_all2.y,2);  
count_epo3=sum(epo_all3.y,2);  


%% setting the number of trials into equal number (88 trials per class)
% 1: setting the number of trials into equal number (88 trials per class)
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


% 2: setting the number of trials into equal number (88 trials per class)
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


% 3: setting the number of trials into equal number (88 trials per class)
y_temp3 = zeros(size(epo_all3.y));

for ci3=1:size(y_temp3,1)
    cidx3 = find(epo_all3.y(ci3,:)==1);
    cidx3 = cidx3(1:88);
    y_temp3(:,cidx3) = epo_all3.y(:,cidx3);
end

idx3 = find(sum(y_temp3(:,:),1)==1);
y3 = y_temp3(:,idx3);
x3 = epo_all3.x(:,:,idx3);

epo_all3.x = x3;
epo_all3.y = y3;

count_epo3=sum(epo_all3.y,2);   


%% tsne for the feature vector distribution

    temp_epo = epo_all;
    temp_epo2 = epo_all2;
    temp_epo3 = epo_all3;
    
   
    %preproc
    temp_epo.x = epo_all.x(:,:,:);
    temp_epo.y = epo_all.y(:,:);
    [fv_tr, csp_w_tr]= proc_multicsp(temp_epo,3);
    fv_tr= proc_variance(fv_tr);        
    fv_tr= proc_logarithm(fv_tr);        
    
    temp_epo2.x = epo_all2.x(:,:,:);
    temp_epo2.y = epo_all2.y(:,:);
    [fv_tr2, csp2_w_tr]= proc_multicsp(temp_epo2,3);
    fv_tr2= proc_variance(fv_tr2);        
    fv_tr2= proc_logarithm(fv_tr2);        
    
    temp_epo3.x = epo_all3.x(:,:,:);
    temp_epo3.y = epo_all3.y(:,:);
    [fv_tr3, csp3_w_tr]= proc_multicsp(temp_epo3,3);
    fv_tr3= proc_variance(fv_tr3);        
    fv_tr3= proc_logarithm(fv_tr3);        
    
    
    %feature vector reshape
    train_size = size(epo_all.x,3); 
    fv_trr = reshape(fv_tr.x,[24,train_size]); %36: feature vector size, 476: number of trials
    
    train_size2 = size(epo_all2.x,3); 
    fv_trr2 = reshape(fv_tr2.x,[24,train_size2]);
    
    train_size3 = size(epo_all3.x,3);
    fv_trr3 = reshape(fv_tr3.x,[24,train_size3]);
    
fv_trr = fv_trr'
fv_trr2 = fv_trr2'
fv_trr3 = fv_trr3'

% y lable (one hot encoding into labels)
species=zeros(1, 352);
species2=zeros(1, 352);
species3=zeros(1, 352);

for uu = 1:352
    
species(1,uu) = find(fv_tr.y(:,uu)==1); % species

end
for uuu = 1:352
    
species2(1,uuu) = find(fv_tr2.y(:,uuu)==1);

end
for uuuu = 1:352
    
species3(1,uuuu) = find(fv_tr3.y(:,uuuu)==1); 

end

species=species';
species2=species2';
species3=species3';

species3 = species3+4;


%% Performing tsne one-by-one
% Y = tsne(fv_trr);

% Y = tsne(add_fv_trr);
% 
% gscatter(Y(:,1),Y(:,2),species)
% 
% meas = fv_trr
% 
% Y = tsne(meas,'Algorithm','exact','Distance','mahalanobis');
% subplot(2,2,1)
% gscatter(Y(:,1),Y(:,2),species)
% title('Mahalanobis')
% 
% 
% Y = tsne(meas,'Algorithm','exact','Distance','cosine');
% subplot(2,2,2)
% gscatter(Y(:,1),Y(:,2),species)
% title('Cosine')
% 
% 
% Y = tsne(meas,'Algorithm','exact','Distance','chebychev');
% subplot(2,2,3)
% gscatter(Y(:,1),Y(:,2),species)
% title('Chebychev')
% 
% 
% Y = tsne(meas,'Algorithm','exact','Distance','euclidean');
% subplot(2,2,4)
% gscatter(Y(:,1),Y(:,2),species)
% title('Euclidean')
% 
% [Y,loss] = tsne(meas,'Algorithm','exact');
% 
% [Y2,loss2] = tsne(meas,'Algorithm','exact','NumDimensions',3);
% 
% fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n',loss,loss2)
% 
% 
% gscatter(Y(:,1),Y(:,2),species,eye(3))
% title('2-D Embedding')
% 
% 
% figure
% v = double(categorical(species));
% 
% % eee = 1:numel(v);
% % eee = eee'
% % 
% % c = full(sparse(eee,v,ones(size(v)),numel(v),3));
% 
% scatter3(Y2(:,1),Y2(:,2),Y2(:,3),15,v,'filled')
% title('3-D Embedding')
% view(-50,8)

%% plotting tsne of 1(imagined speech) and 3(overt speech) of identical classes

% add 1 and 3
add_fv_trr = zeros(704,24);
add_fv_trr(1:352, 1:24) = fv_trr;
add_fv_trr(353:704, 1:24) = fv_trr3;

add_species = zeros(704,1);
add_species(1:352, 1) = species;
add_species(353:704, 1) = species3;

% Y = tsne(add_fv_trr);
% gscatter(Y(:,1),Y(:,2),add_species)


%%

% in order not to change
fv_trr = add_fv_trr;
species = add_species;

meas = fv_trr

% plotting the in four different distances
Y = tsne(meas,'Algorithm','exact','Distance','mahalanobis');
subplot(2,2,1)
gscatter(Y(:,1),Y(:,2),species)
title('Mahalanobis')

Y = tsne(meas,'Algorithm','exact','Distance','cosine');
subplot(2,2,2)
gscatter(Y(:,1),Y(:,2),species)
title('Cosine')

Y = tsne(meas,'Algorithm','exact','Distance','chebychev');
subplot(2,2,3)
gscatter(Y(:,1),Y(:,2),species)
title('Chebychev')

Y = tsne(meas,'Algorithm','exact','Distance','euclidean');
subplot(2,2,4)
gscatter(Y(:,1),Y(:,2),species)
title('Euclidean')

[Y,loss] = tsne(meas,'Algorithm','exact');

[Y2,loss2] = tsne(meas,'Algorithm','exact','NumDimensions',3);

fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n',loss,loss2)

%% 2D plotting, 3D plotting

% figure
% gscatter(Y(:,1),Y(:,2),species,eye(3))
% title('2-D Embedding')

figure
v = double(categorical(species));
% eee = 1:numel(v);
% eee = eee'
 
% c = full(sparse(eee,v,ones(size(v)),numel(v),3));
scatter3(Y2(:,1),Y2(:,2),Y2(:,3),15,v,'filled')
title('3-D Embedding')
view(-50,8)

% [R, P] = corrcoef(csp_w(13,[1,26]), csp_w2(13,[1,26]), csp_w3(13,[1,26]), 'alpha', 0.05);


%% distances between the mean of the classes

% dividing into each classes

id1 = find(species==1);
id2 = find(species==2);
id3 = find(species==3);
id4 = find(species==4);
id5 = find(species==5);
id6 = find(species==6);
id7 = find(species==7);
id8 = find(species==8);

c1 = Y(id1,:);
c2 = Y(id2,:);
c3 = Y(id3,:);
c4 = Y(id4,:);
c5 = Y(id5,:);
c6 = Y(id6,:);
c7 = Y(id7,:);
c8 = Y(id8,:);

mc1 = mean(c1);
mc2 = mean(c2);
mc3 = mean(c3);
mc4 = mean(c4);

mc5 = mean(c5);
mc6 = mean(c6);
mc7 = mean(c7);
mc8 = mean(c8);

distancee_square = (mc1(1,1)-mc5(1,1))^2 + (mc1(1,2)-mc5(1,2))^2 


