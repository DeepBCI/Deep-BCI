%% CCA based classifier
AUC_all = []; mean_AUC=[];

%% Generating Compared Signals
freq = [11 7 5];  % [11 7 5] 5.45, 8.75, 12  [17 11 7 5]
fs = 100;
window_time = 4;

t = [1/fs:1/fs:window_time];

% ground truth
Y=cell(1);
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end

%% Selected channels
chan = {'PO3','POz','PO4','O1','Oz','O2'};

%%
for subNum = 1:17

for ispeed = 1:sum(~cellfun('isempty', EPO_all(subNum,:)))
%% channel select
epo = EPO_all{subNum,ispeed};
epo = proc_selectChannels(epo, chan);

%% one-hot decoding
epo.y_dec = double(onehotdecode(epo.y,[1,2,3],1));

%% accuracy
% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

nTrial = size(epo.y,2);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    [r_value(i), pred(i)]=max(r(i,:));
end
acc=length(find(epo.y_dec == pred))/nTrial;

% Get Accuracy
AUC_all(ispeed,subNum)=acc;

end
end
%% Average Accuracy per Speed
disp('Mean AUC')
for ispeed = 1:4
mean_AUC(ispeed,1) = sum(AUC_all(ispeed,:))/nnz(AUC_all(ispeed,:));
end
disp(mean_AUC)

