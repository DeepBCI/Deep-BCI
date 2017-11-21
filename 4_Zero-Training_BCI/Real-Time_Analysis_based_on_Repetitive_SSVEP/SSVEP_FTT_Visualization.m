% Visualization toolbox for Steady State Visually Evoked Potentials (SSVEP)
% using Fast Fourier Transform

clear all; clc; 

%% EEG file path
dataPath = 'D:\BCICenter\SUBJECT\SESSION'; % Write your own data path
filename = 'ssvep'; %% Write your own data file name

file = fullfile(dataPath, filename);

%% Load EEG file
marker = {'1', 'Class 1'; '2', 'Class 2'; '3', 'Class 3'; '4', 'Class 4'};
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;});

%% Pre-processing the EEG file
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_selectChannels(cnt, {'Name', 'Oz'});
cnt=prep_filter(cnt, {'frequency', [1 40]});
smt=prep_segmentation(cnt, {'interval', [0 4000]});

%% Averaging the Segmented data
avgSMT = prep_average(smt);

%% Visualization
fig = figure;
classNum = size(avgSMT.class, 1);
legend_txt = cell(classNum, 1);
for i = 1:classNum
    [YfreqDomain, freqRange] = positiveFFT(avgSMT.x(:, i), avgSMT.fs);
    plot(freqRange, abs(YfreqDomain));
    legend_txt{i} = smt.class{i,2};
    hold on;
end
legend(legend_txt{:});

%% Option
ax = findobj(fig, 'Type', 'axes');
set(ax, 'Xlim', [0 30]);