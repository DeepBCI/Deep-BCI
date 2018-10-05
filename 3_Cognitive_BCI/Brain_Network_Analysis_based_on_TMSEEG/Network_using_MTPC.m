% This code used Multi-threshold Permutation Correction (MTPC)

%% data re-construction

clear

cd 'H:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE'
folders = dir('H:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE');
folders = {folders.name};
folders = folders(3:end); 

Time = [-1000:2.7586:1000];
rTime = Time(1,344:507);

for i = 1:length(folders)

    cd 'F:\3_TMSEEG_UWdata\2_AnalysisData\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE'

    curFile = folders{i};
    load(curFile);
        
    CM(:,:,i) = mean(Matrix(26:end,:,:),1); % time points * channels * channels
    clear Matrix
end

%% MTPC 

configN.thresh_type = 'density';
configN.thresh = [0.05:0.05:1];

% global? local?
[GT_Ndata,configN] =  MTPC_generate_metrics_global(CM) % CM_all: channels * channels * subjects

% Processing matrix 248 (the num. of subjects)

GT_raw_data = GT_Ndata; % save original data

% Statistics with MTPC 
design = [0;0;0;0;0;0;1;1;1;1;1;1];
configN.thresh = thresh;
configN.modeltype = 'ranksum';
[statsN,configN] = MTPC_evaluate_metrics(GT_Ndata,design,configN)


save PLV2gamma4_100_re8.mat GT_Ndata configN statsN GT_raw_data

