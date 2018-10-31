%% kh_detect_PAC_comodulation.m
%
% It calculates PAC accross time (not time trace) for
%  comouldation matrix (e.g. [low1 ~ low2] x [high1 ~ high2])
% This code includes those steps:
%  1. Band-pass filter raw signals into
%     low freq. (LF) and high freq. (HF)
%  2. Obtain the phase of LF using Hilbert transform (angle)
%  3. Obtain the amplitude of HF using Hilbert transform (abs)
%  4. Compose LF phase and HF amplitude as following:
%     Analytic signal Z = A_HF * exp(1j * P_LF)
%  5. PACs for each trial -> trial average
%
% Mearsuring the intensity of PAC
%  1. from phase sorted plot
%   - KL distance from uniform distribution - MI invented by Tort et al.,
%   2010
%  2. MVL (mean vector length) - MI invented by Canolty et al., 2006
%  3. normalized MVL - MI invented by Ozkurt et al., 2011
%  4. PLV (phase locking value) - MI invented by Cohen et al., 2008
%
% Usage
%   inputs:
%          EEG: data structure that contains information
%             EEG.dat: [1 x time x trials]
%             EEG.time: [1 x time x trials]
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
% For details of the PAC methods go to:
% http://jn.physiology.org/content/104/2/1195.short
% http://science.sciencemag.org/content/313/5793/1626.long
% http://www.sciencedirect.com/science/article/pii/S0165027011004730
% http://www.sciencedirect.com/science/article/pii/S0165027007005237
%
% Please note that this code was modified, user-friendly re-coded, and added
% custom-built statistics from the original code written and maintained 
% by Robert Seymour, June 2017 for verifying this script.
% 
% See https://github.com/neurofractal/sensory_PAC
% 
% CAUTION: This script does not contain data pre-processing steps, so users
% should pre-process data before using the script.
% Reprocued by: Kyungho Won (MEG source -> EEG sensor level) Oct. 2018.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MI_matrix_raw, MI_matrix_surr, MI_matrix_permute] = kh_detect_PAC_comodulation(EEG, f_p, f_a, approach, frame)
% Determine final matrix size
phase_len = length(f_p(1):1:f_p(2)); % LF - 1Hz step
amp_len = length(f_a(1):2:f_a(2)); % HF - 2Hz step

% Create comodulogram matrix
MI_matrix_raw = zeros(amp_len, phase_len);
MI_matrix_surr = zeros(amp_len, phase_len);
MI_matrix_permute = [];
clear phase_len amp_len

row1 = 1; row2 = 1;
% ---------------- test 0910: openSW-friendly test
for lf_phase = f_p(1):1:f_p(2)
    for hf_amp = f_a(1):2:f_a(2)
        % Bandpass filter individual trials using a two-way zero phase lag
        % IIR filter - user can modify
        % Hilbert transform (phase & amplitude envelope)
        % Specifiy bandwith = +- center frequency/2.5
        Af1 = round(hf_amp -(hf_amp/2.5));
        Af2 = round(hf_amp +(hf_amp/2.5));
        A_HF = [];  % amplitude of high freq
        P_LF = []; % phase of low freq
        ampenv_frame = [];
        phase_frmae = [];
        PAC_over_trial = [];
        for nbwindow = 1:size(EEG.data, 3)
            A_HF = cat(1, A_HF, ...
                ft_preproc_bandpassfilter(squeeze(EEG.data(:, :, nbwindow)), EEG.srate, [Af1 Af2], 4, 'but'));
            P_LF = cat(1, P_LF, ...
                ft_preproc_bandpassfilter(squeeze(EEG.data(:, :, nbwindow)), EEG.srate, [lf_phase-1 lf_phase+1], 4, 'but'));
            
            % get amplitude envelop and phase from filtered signal
            % for only current A_HF and P_LF
            ampenv_hilbert = abs(hilbert(A_HF(end, :)));
            phase_hilbert = angle(hilbert(P_LF(end, :)));
            
            current_time = EEG.time(:, :, 1);
            id_frame = [find(current_time==frame(1)), find(current_time==frame(2))];
            
            ampenv_frame(nbwindow, :) = ampenv_hilbert(:, id_frame(1):id_frame(2));
            phase_frmae(nbwindow, :) = phase_hilbert(:, id_frame(1):id_frame(2));
            
            % compute PAC accroding to 'approach'
            switch approach
                case 'canolty'
                    [MI_win] = kh_PAC_canolty(phase_frmae(end, :), ampenv_frame(end, :));
                case 'ozkurt'
                    [MI_win] = kh_PAC_ozkurt(phase_frmae(end, :), ampenv_frame(end, :));
                case 'PLV'
                    [MI_win] = kh_PAC_PLV(phase_frmae(end, :), ampenv_frame(end, :));
                case 'tort'
                    [MI_win] = kh_PAC_tort(phase_frmae(end, :), ampenv_frame(end, :), 18);
            end
            PAC_over_trial = cat(1, PAC_over_trial, MI_win);
        end
        MI_raw = mean(PAC_over_trial); % trial-average
        
        % permutation test - normalize raw PAC
        % Matrix to hold MI permutations
        MI_surr = [];
        
        % For each permutation
        for p=1:1000
            % Get 2 random trial numbers
            trial_num = randperm(size(EEG.data, 3), 2);
            % Extract phase and amplitude info.
            % for different trials & shuffled phase\
            phase_hilbert = phase_frmae(trial_num(1), :);
            phase_hilbert = phase_hilbert(randperm(length(phase_hilbert)));
            ampenv_hilbert = ampenv_frame(trial_num(2), :);

            % Switch PAC method
            switch approach
                case 'canolty'
                    [MI] = kh_PAC_canolty(phase_hilbert, ampenv_hilbert);
                case 'ozkurt'
                    [MI] = kh_PAC_ozkurt(phase_hilbert, ampenv_hilbert);
                case 'PLV'
                    [MI] = kh_PAC_PLV(phase_hilbert, ampenv_hilbert);
                case 'tort'
                    [MI] = kh_PAC_tort(phase_hilbert, ampenv_hilbert, 18);
            end
            
            % Add this value to all other all other values
            MI_matrix_permute(p, row1, row2) = MI;
            MI_surr(p) = MI;
        end
        
        % Subtract the mean of the permutations from the actual PAC
        % value and add this to the surrogate matrix
        MI_surr_normalized = MI_raw - mean(MI_surr);
        % MI_surr_normalized(abs(MI_surr_normalized)<zval) = 0;
        MI_matrix_surr(row1, row2) = MI_surr_normalized; % single permutation-corrected data
        
        % Calculate the raw MI score (without permutation test) and
        % add to the matrix
        MI_matrix_raw(row1, row2) = MI_raw;
        
        % next amp.
        row1 = row1 + 1;
    end
    % next phase
    row1 = 1;
    row2 = row2 + 1;
end

% Displaying result
figure,
pcolor(f_p(1):1:f_p(2),f_a(1):2:f_a(2),MI_matrix_raw);
colormap(jet)
ylabel('Amplitude (Hz)')
xlabel('Phase (Hz)')
title(approach);
colorbar


end
