%% CCA based classifier
ACC_all = []; mean_AUC=[];

%% Generating Compared Signals
freq = [11 7 5];  % [11 7 5] 5.45, 8.75, 12  [17 11 7 5]
fs = 100;
window_time = 5;

t = [1/fs:1/fs:window_time];

% ground truth
Y=cell(1);
for i=1:size(freq,2)
    Y{i}=[sin(2*pi*60/freq(i)*t);cos(2*pi*60/freq(i)*t);sin(2*pi*2*60/freq(i)*t);cos(2*pi*2*60/freq(i)*t)];
end

%% Selected channels
% chan = {'PO7','PO3','POz','PO4','PO8','O1','Oz','O2'};
chan = {'L1','L2','L4','L5','L6','L7','L9','L10','R1','R2','R4','R5','R7','R8'}; % ear-EEG

%%
for subNum = 1:nSub

for ispeed = 2:sum(~cellfun('isempty', EPO(subNum,:)))+1
%% channel select
epo = EPO{subNum,ispeed};
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
ACC_all(ispeed,subNum)=acc;

end
end
%% Average Accuracy per Speed
disp('Mean AUC')
for ispeed = 2:5
mean_AUC(ispeed,1) = sum(ACC_all(ispeed,:))/nnz(ACC_all(ispeed,:));
end
disp(mean_AUC)

