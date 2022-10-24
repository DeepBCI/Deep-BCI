clear all; close all; clc;

BMI.EEG_DIR=['C:\Users\cvpr\Desktop\BTSdataset\shlee'];

filelist={'s4_lcw_sess1_190226'};
%filelist={'s7_kshreal_sess2_190314'};

file=fullfile(BMI.EEG_DIR,'s4_lcw_sess1_190226');
%file=fullfile(BMI.EEG_DIR,'s7_kshreal_sess2_190314');


marker={'1','imagine_Ambulance';  '2','imagine_Clock';  '3','imagine_Hello';  '4','imagine_Helpme';  '5','imagine_Light';  '6','imagine_Pain';  '7','imagine_Stop';  '8','imagine_Thankyou';  '9','imagine_Toilet';  '10','imagine_TV';  '11','imagine_Water';  '12','imagine_Yes';  '13','imagine_Rest'; ...
'101','cue_Ambulance'; '102','cue_Clock'; '103','cue_Hello'; '104','cue_Helpme'; '105','cue_Light'; '106','cue_Pain';  '107','cue_Stop'; '108','cue_Thankyou';  '109','cue_Toilet'; '110','cue_TV';  '111','cue_Water';  '112','cue_Yes'; '113','cue_Rest'; ...
'17','cross'; '201','totalstart'; '202','totalend'; '211','intervalstart'; '212','intervalend'};

% Load EEG files
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});

field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);

CNT_all_class=prep_selectClass(CNT,{'class',{'imagine_Ambulance','imagine_Clock', 'imagine_Hello', 'imagine_Helpme', 'imagine_Light', 'imagine_Pain', 'imagine_Stop', 'imagine_Thankyou', 'imagine_Toilet', 'imagine_TV', 'imagine_Water', 'imagine_Yes', 'imagine_Rest'}});


%% 
CNT2class=prep_selectClass(CNT_all_class,{'class',{'imagine_Rest','imagine_Ambulance'}});


freq_band = [0.5 40];

CNTfilt =prep_filter(CNT2class , {'frequency', freq_band});

% Segmentation
time_interval = [200 1000];

SMT= prep_segmentation(CNTfilt, {'interval', time_interval});

[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [3]});
ft=func_featureExtraction(SMT, {'feature','logvar'});

%cv³»¿ë
for iter=1:10
CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
        'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','10'
% 'leaveout'
};

[loss]=eval_crossValidation(SMT, CV); % input : eeg, or eeg_epo
Result_iter(1,iter)=1-loss';
end

Mean_CV_result=mean(Result_iter',1)';
