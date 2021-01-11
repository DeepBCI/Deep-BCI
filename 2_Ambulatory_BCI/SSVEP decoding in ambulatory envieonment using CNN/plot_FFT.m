
%% Feature extraction

% Make FFT features
% for sub = 1:10

sub = 10;
y_lim = 6;

% epo = epo_train{sub};
epo = epo_test{sub};

% channel selection
chan = {'PO3','POz','PO4','O1','Oz','O2'};

epo = proc_selectChannels(epo, chan);
epo = proc_selectIval(epo, [0 4000]);  % [0 4000]

epo1 = proc_selectClasses(epo,[1]);
epo2 = proc_selectClasses(epo,[2]);
epo3 = proc_selectClasses(epo,[3]);

% data reshape
dataset{1} = permute(epo1.x, [3,1,2]);
[tr, dp, ch] = size(dataset{1}); % tr: trial, dp: time, ch: channel

dataset{2} = permute(epo2.x, [3,1,2]);
[tr, dp, ch] = size(dataset{2}); % tr: trial, dp: time, ch: channel

dataset{3} = permute(epo3.x, [3,1,2]);
[tr, dp, ch] = size(dataset{3}); % tr: trial, dp: time, ch: channel

%% Fast Fourier Transform (FFT)

for c = 1:3
    X_arr=[]; % make empty array
    for k=1:tr % trials
        x=squeeze(dataset{c}(k, :,:)); % data
        N=length(x);    % get the number of points
        kv=0:N-1;        % create a vector from 0 to N-1
        T=N/epo.fs;         % get the frequency interval
        freq=kv/T;       % create the frequency range
        X=fft(x)/N*2;   % normalize the data
        cutOff = ceil(N/2); % get the only positive frequency

        % take only the first half of the spectrum
        X=abs(X(1:cutOff,:)); % absolute values to cut off
        freq = freq(1:cutOff); % frequency to cut off
        XX = permute(X,[3 1 2]);
        X_arr=[X_arr; XX]; % save in array
    end
    X_class{c} = X_arr;
    avg_class{c} = squeeze(mean(mean(X_arr,3),1));
end


%% frequency band
figure(1)
% f_gt = [11 7 5];  % 5.45, 8.75, 12
plot(freq,avg_class{1})
hold on
plot(freq,avg_class{2})
plot(freq,avg_class{3})

xlim([0 15])

grid on

line([5.45 5.45], [0 y_lim],'Color','k','LineStyle','--')
line([8.75 8.75], [0 y_lim],'Color','k','LineStyle','--')
line([12 12], [0 y_lim],'Color','k','LineStyle','--')
ylim([0 y_lim])
legend('5.45 Hz', '8.75 Hz', '12 Hz')
hold off

% end



