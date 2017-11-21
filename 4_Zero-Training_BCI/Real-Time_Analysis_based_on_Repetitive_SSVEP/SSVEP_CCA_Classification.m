% Classification toolbox for for Steady State Visually Evoked Potentials
% (SSVEP) using Canonical-Correlation Analysis (CCA)

clear all; clc; 

%% EEG file path
dataPath = 'D:\StarlabDB_2nd\subject8_dblee\Session1';
filename = 'ssvep_off';

file = fullfile(dataPath, filename);

%% Load EEG file
marker = {'1', 'Class 1'; '2', 'Class 2'; '3', 'Class 3'; '4', 'Class 4'};
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;});

%% Pre-processing the EEG file
cnt=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
cnt=prep_filter(cnt, {'frequency', [1 40]});
smt=prep_segmentation(cnt, {'interval', [0 4000]});

%% Initializing variables for using CCA;
refFreq = [60/5, 60/7, 60/9, 60/11];
time = 4; % Seconds;
classNum = size(smt.class, 1);
trialNum = size(smt.y_dec, 2);
loss = 0;

t = 0:1/smt.fs:time;

Y = cell(1, classNum);
r = zeros(1, classNum);

for i = 1:classNum
    ref = 2*pi*refFreq(i)*t;
    Y{i} = [sin(ref); cos(ref); sin(ref*2); cos(ref*2)];
end

%% Analysing SSVEP using CCA in single trials
for i = 1:trialNum
    data = squeeze(smt.x(:, i, :));
    for j = 1:classNum
        [~, ~, corr] = canoncorr(data, Y{j}');
        r(j) = max(corr);
    end
    [~, ind] = max(r);
    
    if ~smt.y_logic(ind, i)
        loss = loss + 1;
    end
end





