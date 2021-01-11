%% Training
for sub = 1:10

% channel select
epo = epo_train{sub};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, time_interval);  % [0 4000]

% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

% get features
nTrial = length(epo.y_dec);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    [r_value(i), pred(i)]=max(r(i,:));
end

% threshold
rere=r;
r_thres = rere.*(1-epo.y'); % non-target 만 남기기
thres_tr(sub,:) = mean(r_thres);

end

%% Test
excel_ACC = [];

for sub = 1:10
% channel select
epo = epo_test{sub};
epo = proc_selectChannels(epo, chan_cap);
epo = proc_selectIval(epo, time_interval);  % [0 4000]

% initialization
r_corr = []; r=[]; r_value=[]; pred=[];

% feature extraction
nTrial = length(epo.y_dec);
for i=1:nTrial
    r_dump = [];
    for j=1:size(freq,2)
        [~,~, r_corr{j}] = canoncorr(squeeze(epo.x(:,:,i)),Y{j}');
        r_dump = [r_dump mean(r_corr{j})];
    end
    r(i,:) = r_dump;
    % get threshold from training
    r2(i,:) = r(i,:)-thres_tr(sub,:);
    [r_value(i), pred(i)]=max(r2(i,:));
end
acc_all=length(find(epo.y_dec == pred))/nTrial;

excel_ACC(1,sub)=acc_all;
end

% accuracy
disp('Mean ACC')
mean_ACC = sum(excel_ACC(1,:))/nnz(excel_ACC(1,:));

disp(mean_ACC)