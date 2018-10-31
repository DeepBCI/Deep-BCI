% This function makes compatible structure as below
% eeg = 
%   total_n_trials
%   non_P3_n_trials
%   P3_n_trials
%   srate
%   data
%   commnet
%   event
%   target

function y = load_P3BCI2017(filename)
N = 6; % 6 x 6 matrix
reptition_seq = N * 2* 15; % N x N speller, (row+col), 15 repetitions

y = [];
req_events = {'StimulusCode', 'SelectedTarget'};
% keyDown: keystroke | phaseSequence: frame (begin, during, end)

EEG = pop_loadBCI2000({filename}, req_events);
% [EEG, EEG.indelec, measure, com] = ...
%     pop_rejchan(EEG, 'elec', [1:32], 'threshold', 5, 'norm', 'on', 'measure', 'prob');
% EEG = eeg_checkset(EEG);
% indelec - indices of rejected electrodes
% fprintf('%02d was rejected..\n', EEG.indelec);

[signal, states, parameters] = load_bcidat(filename, '-calibrated');
disp(parameters.TextToSpell.Value);

EEG.spellermatrix = ...
    ['A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'];
EEG.event_latency = [EEG.event.latency]; % event latency
EEG.event_pos = [EEG.event.position]; % event type
EEG.event_type = {EEG.event.type}; % event name

% ---- 
tmp_spell_idx = find(strcmp(EEG.event_type, req_events{2}));
EEG.targetText = parameters.TextToSpell.Value{1}; % text to be copied
EEG.textResult = EEG.spellermatrix(EEG.event_pos(tmp_spell_idx));
EEG.event_pos(tmp_spell_idx) = [];
EEG.event_latency(tmp_spell_idx) = [];
EEG.event_type(tmp_spell_idx) = [];

% now event contains only codes
disp(['Answer is ' EEG.textResult]);

% step1. make event
event = zeros(1, size(EEG.data, 2));
for i=1:length(EEG.event_pos);
    event(EEG.event_latency(i)) = EEG.event_pos(i);
end

% step2. make target
target = zeros(1, size(EEG.data,2));
for i=1:length(EEG.targetText)
     tmp_index = find(EEG.spellermatrix==EEG.targetText(i)); % target seq value among 1 ~ 12
     ttmp_col = mod(tmp_index, N);
     if ttmp_col == 0;
         ttmp_col = N;
     end
     ttmp_col = ttmp_col+N; % column: 7~ 12
     ttmp_row = ceil(tmp_index/N); % row: 1 ~ 6
     
     % ---- find out target column and row latency
     idx_t_col = find(EEG.event_pos(1+reptition_seq * (i-1):reptition_seq * i) == ttmp_col) + reptition_seq * (i-1);
     idx_t_row =  find(EEG.event_pos(1+reptition_seq * (i-1):reptition_seq * i) == ttmp_row) + reptition_seq * (i-1);
     % ---- find out nontarget column and row latency
     idx_nt_col_row = find(~ismember(1+reptition_seq * (i-1):reptition_seq * i, union(idx_t_col, idx_t_row))) + reptition_seq * (i-1);   
 
     target(EEG.event_latency(union(idx_t_col, idx_t_row))) = 1;
     target(EEG.event_latency(idx_nt_col_row)) = 2;
end

y.total_n_trials = length(EEG.event_latency);
y.P3_n_trials = size(find(target==1), 2);
y.non_P3_n_trials = size(find(target==2), 2);
y.comment = filename;
y.srate = EEG.srate;
y.event = event;
y.target = target;
y.data = EEG.data;
y.nbchan = EEG.nbchan;
y.res = EEG.targetText;
y.onlineacc = sum(y.res == EEG.textResult) / length(y.res);

end