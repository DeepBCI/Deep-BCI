dataPath = 'D:\BCICenter\SUBJECT\SESSION'; % Write your own data path
filename = 'p300'; %% Write your own data file name

file=fullfile(dataPath,filename);

segTime=[-200 800];
baseTime=[-200 0];
selTime=[0 800];
marker={'1','target';'2','nontarget'};

fs=100; 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});

cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_selectChannels(cnt,{'Name',{'Cz', 'Oz'}});
cnt=prep_filter(cnt, {'frequency', [0.5, 40]});

smt=prep_segmentation(cnt, {'interval', segTime});
smt=prep_baseline(smt, {'Time',baseTime});
smt=prep_selectTime(smt, {'Time',selTime});

plot_x = 0:10:800;
t_c = smt.x(:,smt.y_logic(1,:),1);
t_o = smt.x(:,smt.y_logic(1,:),2);
n_c = smt.x(:,smt.y_logic(2,:),1);
n_o = smt.x(:,smt.y_logic(2,:),2);
t_c = mean(t_c,2);
t_o = mean(t_o,2);
n_c = mean(n_c,2);
n_o = mean(n_o,2);
f=figure;
y_l=[min([min(t_c), min(t_o),min(n_c), min(n_o)])*1.1 max([max(t_c), max(t_o),max(n_c), max(n_o)])*1.1];
subplot(2,1,1); plot(plot_x, t_c); hold on; plot(plot_x, n_c);
legend('target', 'non-target'); title('Cz'); ylim(y_l);
subplot(2,1,2); plot(plot_x, t_o); hold on; plot(plot_x, n_o);
legend('target', 'non-target'); title('Oz'); ylim(y_l);