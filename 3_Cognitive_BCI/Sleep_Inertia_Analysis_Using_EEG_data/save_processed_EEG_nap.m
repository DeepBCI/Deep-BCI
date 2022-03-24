% ----------------------------------------------------------------------- %
% Run file 'save_processed_EEG_nap.m' pre-processes EEG signal,           %
% especially nap data. (Based on KU-dataset, with EEGlab version 2021.1)  %
% ----------------------------------------------------------------------- %
%   Input parameters: None                                                %
%   Output variables:                                                     %
%       - .mat files: pre-processed EEG data of subjects                  %
%   Notes:                                                                %
%       - Line 19, Line 24, Line 61: you should edit dir to use.          %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:      1.0.                                              %
%       - Author:       hn_jo@korea.ac.kr                                 %
%       - Date:         03/24/2022                                        %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       - edit file dir and f5.                                           %
% ----------------------------------------------------------------------- %
files = dir('C:\'); % edit file dir to use!!
k = length(files);

for file = 1:k
    % data loading
    EEG = pop_loadbv('C:\', files(file).name); % edit file dir to use!!
    
    % file name setting
    EEG.setname='EEG';
    
    % event select 
    EEG = pop_rmdat( EEG, {'S111'},[0 5400] ,0); % 'S111' = start event
    
    % resampling to 250Hz
    EEG = pop_resample(EEG, 250);
    
    % filtering [0.5 50]
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',50,'plotfreqz',1);
    close all;
    
    % set EEG channel location (60 ch, standard 10-20)
    EEG = pop_chanedit(EEG, 'lookup','C:\\eeglab\\eeglab2021.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1020.elc');
    
    % remove EOG channel
    EEG = pop_select( EEG, 'nochannel',{'REOG','LHEOG','UVEOG','LVEOG'});
    
    % segmentation 30s (60 * 7500 * 180) 
    EEG = eeg_regepochs(EEG, 30);

    % find bad channel
    [~, bad_index] = pop_rejchan(EEG, 'elec', [1:EEG.nbchan],'threshold',3,'norm','on','measure','kurt');
    msg = ['bad channel: ' num2str(bad_index)];
    disp(msg)

    % interpolation bad channel
    for bad = bad_index
        EEG = pop_interp(EEG, bad, 'spherical');
        msg = ['Interpolating channel ' num2str(bad)];
        disp(msg)
    end

    % save .mat file
    filename = ['C:\' int2str(file-1) '.mat']; % edit file dir to use!!
    save(filename, 'EEG');
end