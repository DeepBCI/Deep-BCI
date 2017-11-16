% Event-related (de)synchronization
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
% Hilbert transform
TRIN_ENV=prep_envelope(TRIN);
% Segmentation
interval=[-500 4990];
ENV_EPO=prep_segmentation(TRIN_ENV, {'interval', interval});
% Baseline correction
ENV_EPO=prep_baseline(ENV_EPO,{'Time',[-490 0]});
% Class selection
Cls1=prep_selectClass(ENV_EPO,{'class','right'});
Cls2=prep_selectClass(ENV_EPO,{'class','left'});
% Average
Cls1.x=squeeze(mean(Cls1.x,2));
Cls2.x=squeeze(mean(Cls2.x,2));
left_chan=18; % Channel C3
right_chan=51; % channel C4
xx=-500:10:4990;

figure % Class1: right, Class2: foot
subplot(1,2,1)
plot(xx,Cls1.x(:,left_chan)/100,'r-',xx,Cls2.x(:,left_chan)/100,'b--'); % C4
xlim([-500 4990]); %xlabel('Time (ms)')
% ylim([-50/100 10/100]); 
ylabel('Amplitude (uV)');
title('C3 channel') % Left (hemi-sphere)
legend('Right','Foot')

subplot(1,2,2)
plot(xx,Cls1.x(:,right_chan)/100,'r-',xx,Cls2.x(:,right_chan)/100,'b--'); % C4
xlim([-500 4990]); %xlabel('Time (ms)')
% ylim([-50/100 10/100]); 
ylabel('amplitude (uV)')
title('Cz channel') % Right (hemi-sphere)
legend('Right','Foot')



