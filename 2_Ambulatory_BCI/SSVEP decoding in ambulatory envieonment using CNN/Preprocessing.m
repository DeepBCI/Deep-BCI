clear all;
cd C:\Users\yelee\Desktop\test_tc2_4
cd new_BBCI_toolbox 
startup_bbci_toolbox
cd ..
%% Data Load
dire = 'data';

for sub=1:10
    filename = sprintf('s%d',sub);
    epo_train{sub} = load(fullfile(dire,'train',filename)).epo;
    epo_test{sub} = load(fullfile(dire,'test',filename)).epo;
end
cd TC4
%% CCA based classifier setting

chan_cap = {'Oz'};
time_interval = [1002 2000];

freq = [11 7 5];  % [11 7 5]    5.45, 8.75, 12 
fs = 500;
window_time = 1;
%(interval_sub(sub,2)-interval_sub(sub,1))/1000;

t = [1/fs:1/fs:window_time];

% ground truth
Y=cell(1);
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end
