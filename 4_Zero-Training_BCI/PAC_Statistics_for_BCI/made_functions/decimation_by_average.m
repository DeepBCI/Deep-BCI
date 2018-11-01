% ------ replacing each sequence of 24 samples with their mean
% Fs:512, Frame: [0 800]ms -> 17 samples (Krusienski et al., 2006)
function y = decimation_by_average(data, factor, len_trials)

% data type: ch x frame (maybe trigger-extracted data)
n_sample = factor;

frame_trial = length(data)/len_trials; % single trial frame
% disp(frame_trial/n_sample);
% disp(len_trials);

for deci=1:len_trials
    single_epoch = data(:, frame_trial * (deci-1) +1:frame_trial *deci);
    
    for avg_deci=1:floor(frame_trial/n_sample)
        y(:, floor(frame_trial/n_sample) * (deci-1) + avg_deci) = mean(single_epoch(:, (avg_deci-1)*n_sample+1:avg_deci *n_sample), 2);
    end
end



% disp(size(y));
end