%% extract_event_3D.m
%
% extract_event_3D(): returns data structure [ch x time x trials]
% Usage:
%  input() - data [ch x time] 
%          - event [1 x time]: 1 at event, 0 otherwise
%          - srate: sampling rate
%          - frame: [start end] in ms
%
%  output() - data [ch x time x trials]
function dout = extract_event_3D(din, event, srate, frame)
id_events = find(event ~= 0);
nevents = length(id_events);
dout = [];
for iter_tr = 1:nevents
    ind = event(id_events(iter_tr));
    interval = ind + round(frame(1) * srate/1000):ind+round(frame(2)*srate/1000);
    tmp_dat = din(:, interval);
    
    dout = cat(3, dout, tmp_dat);
end
end