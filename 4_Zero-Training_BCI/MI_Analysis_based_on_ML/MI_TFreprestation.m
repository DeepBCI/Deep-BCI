% Time-Frequency representation
clc
clear
PR_BCI('C:\Users\cvpr\Desktop\DeepBCI\github\Center') % Edit the variable BMI if necessary
Dirt='C:\Users\cvpr\Desktop\DeepBCI\github\Center\Data';
%% Load
file=fullfile(Dirt, '\MI_demo_tr');
marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_DAT(file,{'device','brainVision';'marker', marker;'fs', [100]});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
TRIN=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
TRIN=prep_selectClass(TRIN,{'class',{'right', 'left'}});

%%
% BPF
band=[8 13];
TRIN=prep_filter(TRIN, {'frequency', band});
% Segmentation
interval=[-500 3990];
EPO=prep_segmentation(TRIN, {'interval', interval});
% Class selection
Cls1=prep_selectClass(EPO,{'class','right'});
Cls2=prep_selectClass(EPO,{'class','left'});
%%STFT spectrogram
N_trial=4;
Chan1=18;
Chan2=51;
x1=Cls1.x(:,N_trial,Chan1); % point * trial * chan, 귀: right class, 
x2=Cls1.x(:,N_trial,Chan2); % point * trial * chan, 운동: right class, 9번 C3 채널 

fs = 100;
R = 50;
window = hamming(R);
N = 2^16;
L = ceil(R*0.5);
overlap = R-L;

% x1=tm1;
[s f t]=spectrogram(x1,window,overlap,N,fs,'yaxis');
[s2 f2 t2]=spectrogram(x2,window,overlap,N,fs,'yaxis');

figure, clf
% imagesc(t,f,log10(abs(s)));
subplot(1,2,2)
imagesc(t-0.5,f,sqrt(abs(s/400)));
colormap(jet)
% caxis([0 2.2])
% yaxis([0 30])
axis xy
xlabel('time')
ylabel('frequency')
title('C3 channel')
% title('spectrogram, R=50')
colorbar

% figure
subplot(1,2,1)
imagesc(t2-0.5,f2,sqrt(abs(s2/400)));
colormap(jet)
caxis([0 2.2])
% yaxis([0 30])
axis xy
xlabel('time')
ylabel('frequency')
title('C4 channel')
% title('spectrogram, R=50')
colorbar