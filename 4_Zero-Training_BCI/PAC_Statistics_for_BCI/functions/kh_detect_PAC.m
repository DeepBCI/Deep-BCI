%% kh_detect_PAC.m
%
% It calculates PAC accross time (not time trace) for
%  representative band (e.g. theta - low gamma)
% This code includes those steps:
%  1. Band-pass filter raw signals into
%     low freq. (LF) and high freq. (HF)
%  2. Obtain the phase of LF using Hilbert transform (angle)
%  3. Obtain the amplitude of HF using Hilbert transform (abs)
%  4. Compose LF phase and HF amplitude as following:
%     Analytic signal Z = A_HF * exp(1j * P_LF)
%  5. PACs for each trial -> trial average
%
% Checking the existence of PAC:
%  1. Draw {P_LF, A_HF} on the complex plane
%   - asymmetry: PAC exits, symmetry: no PAC
%  2. Draw phase sorted ampltiude plot:
%   - set the number of phase bins (N)
%   - generate histogram with phase bins and correponding amplitude
%   - fluctuation: PAC exists, symmetry: no PAC (1/N)
%
% Mearsuring the intensity of PAC
%  1. from phase sorted plot
%   - KL distance from uniform distribution - MI invented by Tort et al.,
%   2010
%  2. MVL (mean vector length) - MI invented by Canolty et al., 2006
%  3. normalized MVL - MI invented by Ozkurt et al., 2011
%  4. PLV (phase locking value) - MI invented by Cohen et al., 2008

% Statistical test - nonparametric permutation test
%  1. original data
%   : trial 1: [P_LF1 A_HF1] -> PAC 1
%     trail 2: [P_LF2 A_HF2] -> PAC 2
%     ...
%  2. randomization (e.g. 1,000 permutation)
%   : trial shuffling (eg. P_LF1 A_HFj -> PAC s1
%   ...
%  3. get data distribution from step 2
%  4. calculate p-value (uncorrected)

% Multiple comparision test
%  1. options - bonferrnoi, FDR, cluster(based on size and t-value), min-max
%  2. for each permutation, store its max. and min., etc.
%  3. get permutation distribution
%  4. obtain significant threshold
%
% Dependencies
%  : it uses EEGLAB and FiledTrip opensource toolbox for bandpass filtering 
%  and dispalying scalp topography
%
% Usage
%   inputs:
%          EEG: data structure that contains information
%             EEG.dat: [ch x time x trials]
%             EEG.srate: sampling rate
%
%          f_p: frequency of phase-modulated (low freq) [f_p1 f_p2]
%          f_a: frequency of amplitude_modulated (high freq) [f_a1 f_a2]
%          approach: 'canolty', 'ozkurt', 'PLV', 'tort'
%          alpha: significance level (default: 0.05)
%
%   outputs:
%          out_raw_PAC: raw PAC
%          out_corrected_PAC: thresholded PAC with alpha
%          permap_PAC: permutation distribution for each channel [ch x 1000]
%
function [out_raw_PAC, out_corrected_PAC, permap_PAC] = kh_detect_PAC(EEG, f_p, f_a, approach, alpha)

% when alpha is not assigned, use alpha = 0.05
if nargin < 5
    alpha = 0.05;
end
% Compute PAC for each channel
% User may select specific channels by changing "target_ch" below 
target_ch = size(EEG.dat, 1);
out_raw_PAC = [];
out_corrected_PAC =[];
permap_PAC = [];
raw_PAC = []; % PAC for each channel
for nbch = 1:target_ch
    
    A_HF = []; % amplitude of high freq
    P_LF = []; % phase of low freq
    % Compute PAC accross trial(window) level
    PAC_over_trial = [];
    for nbwindow = 1:size(EEG.dat, 3)
        % store A_HF and P_LF for later use
        A_HF = cat(1, A_HF, ...
            ft_preproc_bandpassfilter(squeeze(EEG.dat(nbch, :, nbwindow)), EEG.srate, f_a, [], 'fir')); 
        P_LF = cat(1, P_LF, ...
            ft_preproc_bandpassfilter(squeeze(EEG.dat(nbch, :, nbwindow)), EEG.srate, f_p, [], 'fir'));
        
        % get amplitude enveolpe and phase from filtered signal
        % but only use current A_HF and P_LF
        ampenv_hilbert = abs(hilbert(A_HF(end, :))); 
        phase_hilbert = angle(hilbert(P_LF(end, :)));
        
        % compute PAC accroding to 'approach'
        switch approach
            case 'canolty'
                [MI_win] = kh_PAC_canolty(phase_hilbert, ampenv_hilbert);
            case 'ozkurt'
                [MI_win] = kh_PAC_ozkurt(phase_hilbert, ampenv_hilbert);
            case 'PLV'
                [MI_win] = kh_PAC_PLV(phase_hilbert, ampenv_hilbert);
            case 'tort'
                [MI_win] = kh_PAC_tort(phase_hilbert, ampenv_hilbert, 18);
        end
        PAC_over_trial = cat(1, PAC_over_trial, MI_win);        
    end
    raw_PAC(nbch) = mean(PAC_over_trial); % trial-averge
    
    % permutation test - normalize raw PAC
    permute_PAC = []; % obtain permutation distribution of PAC
    % parallel for loop 
    %  use 'for' instead if you don't have paraell toolbox
    parfor p=1:1000
        fprintf('%4d/1000 permutations...\n', p);
        % trial(window) shuffling of low freq
        % user can chagne what to shffle accroding to data characteristic
        shuffle_id = randperm(size(P_LF, 1));
        permute_P_LF = P_LF(shuffle_id, :);
        
        window_MI = [];
        for nbwindow = 1:size(P_LF, 1)
            ampenv_p = abs(hilbert(A_HF(nbwindow, :)));
            phase_p = angle(hilbert(permute_P_LF(nbwindow, :)));
            
            switch approach
                case 'canolty'
                    [tmp_MI] = kh_PAC_canolty(phase_p, ampenv_p, 18);
                case 'ozkurt'
                    [tmp_MI] = kh_PAC_ozkurt(phase_p, ampenv_p);
                case 'PLV'
                    [tmp_MI] = kh_PAC_PLV(phase_p, ampenv_p);
                case 'tort'
                    [tmp_MI] = kh_PAC_tort(phase_p, ampenv_p);
            end
           window_MI(nbwindow) = tmp_MI; 
        end
        % get PAC distribution from permutation
        permute_PAC(p) = mean(window_MI);
    end
    permap_PAC(nbch, :) = permute_PAC; % save distribution for each channel
    zval = kh_p2zval(alpha);
    threshold = zval * std(permute_PAC) + mean(permute_PAC);
    disp(threshold);
    % observe raw_PAC among PAC distribution generated from permutations
    % turn on only if you need (by un-commenting)  
%     figure, hist(permute_PAC); hold on;
%     title('Chnannel %02d - PAC', nbch);
%     p1 = plot([raw_PAC(target_ch) raw_PAC(target_ch)], get(gca, 'Ylim')); 
%     p2 = plot([threshold threshold], get(gca, 'Ylim'), 'r:');
%     legend([p1 p2], {'raw PAC', 'threshold'});
    
    out_raw_PAC(nbch) = raw_PAC(nbch);
    if raw_PAC(nbch) < threshold
        raw_PAC(nbch) = 0;
    end
    out_corrected_PAC(nbch) = raw_PAC(nbch);
    
end

end