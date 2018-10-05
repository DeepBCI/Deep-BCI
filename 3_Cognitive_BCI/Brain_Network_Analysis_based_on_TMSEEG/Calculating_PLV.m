%% Calculation of PLV between two signals %%
clear
close

folders = dir('F:\TMSEEG\analysis(Jaakko_trials)\2_mat_format\NCE');
folders = {folders.name};

thre = [0:0.01:0.5];

for i = 1:length(folders)
    
    %% Data load
   
    curFile = folders{i};
    load(curFile);
    
    eegData = Mean_Final(:,344:507,:); % channel * samples(-50 ~ 400 msec) * trials
        
    
    %% Set to Parameter

    srate = 362.5; % sampling rate
    fmin = 30; % frequency - low
    fmax = 40; % frequency - high
    
    %% Step2: connectivity measure (eg. PLV)

    plv = PLV(eegData,srate,fmin,fmax);
    RawMatrix = plv + permute(plv,[1 3 2]);
    Matrix = plv + permute(plv,[1 3 2]); % samples * channels * channels
    
    fname = ['Matrix_',curFile];
    save(fname, 'Matrix');
    %
    %% Step 3: Thresholding
    
        clear buffer
        nChannel = size(Matrix,2);
        threMatrix = zeros(length(Matrix),nChannel,nChannel);
    
        for k = 1:length(thre)
            p = thre(k); % proportion for thresholding
            for m = 1:length(Matrix)
                buffer = squeeze(Matrix(m,:,:));
                threMatrix(m,:,:) = threshold_proportional(buffer, p);
            end
            AllMatrix(:,:,:,k) = threMatrix(:,:,:);
        end
    
        fname = ['Thre_se_',curFile];
        save(fname, 'AllMatrix');
    
end

