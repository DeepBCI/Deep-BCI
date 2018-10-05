clear 

%% Data load

cd 'F:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE'
folders = dir('F:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE');
folders = {folders.name};
folders = folders(3:end);

for i=1:6 % CE
    
    cd 'F:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE'
    curFile = folders{i};
    load(curFile);
    
    disp(['Calculating on ' curFile ]);
   
    meanMatrix = squeeze(mean(Matrix(26:end,:,:),1)); % 24:end or 86:111
  
    CE(:,:,i) = meanMatrix;
   
end

j = 1;
for i=7:12 % NCE
    
    cd 'F:\3_TMSEEG_UWdata\2_AnalysisData_NetworkAnalysis\3_analysis(Jaakko_trials)\4_PLV_sub_fail\4_PLV_100_re1\gamma4_CE'
    curFile = folders{i};
    load(curFile);
    
    disp(['Calculating on ' curFile ]);
       
    meanMatrix = squeeze(mean(Matrix(26:end,:,:),1)); 

    
    NCE(:,:,j) = meanMatrix;
    j = j+1;
end

nCE = CE;
nNCE = NCE;


%% statistics

alpha = .05;

for i = 1:60
    for j = 1:60        
        NnCE = nCE(i,j,:);
        NnNCE = nNCE(i,j,:);
        data = { NnCE NnNCE };
        [t(i,j),df,pvals(i,j)] = statcond(data,'method','perm','naccu',1000);
 
    end
end

[p_fdr, p_masked] = fdr(pvals, alpha);
% [p_fdr_bh, p_masked_bh] = bonf_holm(pvals, alpha); % Bonferroni-Holm correction

figure()
imagesc(p_masked)

% plot for p-values matrix

pvals1 = pvals;
pvals2 = pvals;

for i = 1:60
    for j = 1:60        
        if i == j
            pvals1(i,j) = 1; 
            pvals2(i,j) = 0; 
    end
    end
end


clims = [0 1]; 

figure()
subplot(2,1,1); imagesc(pvals1, clims); colormap(jet); colorbar; xlabel('Channel'); ylabel('Channel'); 
subplot(2,1,2); imagesc(pvals2.*p_masked); colormap(jet); colorbar; xlabel('Channel'); ylabel('Channel'); 

%% re-organization for network

CE_Mat = mean(nCE,3).*p_masked;
NCE_Mat = mean(nNCE,3).*p_masked;
t_Mat = t.*p_masked;

CE_ijw = adj2edgeL(triu(CE_Mat));             % passing from matrix form to edge list form
NCE_ijw = adj2edgeL(triu(NCE_Mat));             % passing from matrix form to edge list form
t_ijw = adj2edgeL(triu(t_Mat));             % passing from matrix form to edge list form

range = t_ijw(:,3);
min(range)
max(range)

%% figure using "topoplot_connect"

CE_ds.chanPairs = CE_ijw(:,1:2);
CE_ds.connectStrength = CE_ijw(:,3);
CE_ds.connectStrengthLimits = [0.3 0.7];

NCE_ds.chanPairs = NCE_ijw(:,1:2);
NCE_ds.connectStrength = NCE_ijw(:,3);
NCE_ds.connectStrengthLimits = [0.3 0.7];

t_ds.chanPairs = t_ijw(:,1:2);
t_ds.connectStrength = t_ijw(:,3);
t_ds.connectStrengthLimits = [-5 5];

figure(); colormap('jet');
subplot(1, 3, 1); topoplot_connect(CE_ds, poslocs);
subplot(1, 3, 2); topoplot_connect(NCE_ds, poslocs);
subplot(1, 3, 3); topoplot_connect(t_ds, poslocs);


