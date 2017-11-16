clc
clear
PR_BCI('C:\Users\cvpr\Desktop\DeepBCI\github\Center') % Edit the variable BMI if necessary
Dirt='C:\Users\cvpr\Desktop\DeepBCI\github\Center\Data';
%% Training
% Load
file=fullfile(Dirt, '\MI_demo_tr');
marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_DAT(file,{'device','brainVision';'marker', marker;'fs', [100]});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
TRIN=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
TRIN=prep_selectClass(TRIN,{'class',{'right', 'left'}});
% Preprocessing
TRIN=prep_filter(TRIN, {'frequency', [7 13]});
TRIN_SEG=prep_segmentation(TRIN, {'interval', [750 3500]});
% Feature Extraction
TRIN_FEAT=func_featureExtraction(TRIN_SEG, {'feature','logvar'});
% Classification
[CF_PARAM]=func_train(TRIN_FEAT,{'classifier','LDA'});
%% Test
% Load
file=fullfile(Dirt, '\MI_demo_te');
marker={'1','left';'2','right';'3','foot'};
[EEG.data, EEG.marker, EEG.info]=Load_DAT(file,{'device','brainVision';'marker', marker;'fs', [100]});
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
TEST=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
TEST=prep_selectClass(TEST,{'class',{'right', 'left'}});
% Preprocessing
TEST=prep_filter(TEST, {'frequency', [7 13]});
TEST_SEG=prep_segmentation(TEST, {'interval', [750 3500]});
% Feature Extraction
TEST_FEAT=func_featureExtraction(TEST_SEG, {'feature','logvar'});
% Classification
OUT=func_predict(TEST_FEAT, CF_PARAM);
%% Evaluation
ACCURACY=func_accu(TEST_FEAT.y_dec, OUT);
















