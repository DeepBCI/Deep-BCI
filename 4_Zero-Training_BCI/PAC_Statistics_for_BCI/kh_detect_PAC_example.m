%% Loading eyes-closed resting state data (sample)
clear; clc; close all;
load('biosemi32_locs.mat'); % topographic location file (or .locs)
ch = 1:32;
band_hf = [30 60];
band_lf = [8 13];
sub_corrected_PAC = [];
for nsb=1
    
    try
        filename = sprintf('./samples_detection/S%02d/REST_CLOSES001R01.dat', nsb);
        EEG = pop_loadBCI2000({filename}, {});
        EEG.chanlocs = biosemi32_locs;
    catch
        warning('No such file or directory');
    end
    
    EEG.exg = EEG.data(ch(end)+1:end, :); % extra channels, ECG, EOG, EMG
    EEG.data = EEG.data(ch,:);
    EEG.data = reref(EEG.data(1:32, :), [], 'keepref', 'on');
    EEG.nbchan = length(ch);
      
    % There is no event in sample data so we segment data with 
    % custom window length = 10s.
    window_PAC = 10:10:130; % 24 windows (sample)
    Cliped_dat = EEG.data(:, EEG.srate * window_PAC(1) +1:EEG.srate * window_PAC(end));
    segmented3D_dat = reshape(Cliped_dat, 32, length(Cliped_dat) / (length(window_PAC)-1), ...
        length(window_PAC)-1);
    % output dimension: [ch x time x trial(window)]
    EEG_pac = [];
    EEG_pac.dat = double(segmented3D_dat);
    EEG_pac.srate = EEG.srate;
    
    [raw_PAC, corrected_PAC, permap_PAC] = kh_detect_PAC(EEG_pac, band_lf, band_hf, 'ozkurt', 0.05);
    disp('done.');
    
    figure, topoplot(corrected_PAC, biosemi32_locs, 'electrodes', 'ptslabels', ...
        'numcontour', 0);
    colorbar; caxis([0 max(corrected_PAC)]); colormap(flipud(hot));
    title(sprintf('s%02d', nsb), 'fontsize', 14);
    sub_corrected_PAC = cat(1, sub_corrected_PAC, corrected_PAC);
    
      
end