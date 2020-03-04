% ce_at_LOC.m
%
% Calculation effect site concentration at loss of consciousness
%
% author: Young-Seok Kweon
% created: 2020.02.09
%% init
clc; clear; close all;
%% load 
type='PPF';
% type='MDZ';
num_f=5; % LOC가 되려면 몇번 연속으로 실패해야하나
for i=1:30
    x=load(['latency\',type,'_S',num2str(i)]);
    y=load(['ce\',type,'_S',num2str(i)]);
    ce=y.ce;
    time_ce=y.time_ce;
    label=x.label_new;
    label_tag=label.label;
    latency=label.latency;
    s1=label.start_time_b;
    s2=label.start_time_s;
    for j=1:length(label_tag)-num_f
        if label_tag(j)==1 && sum(label_tag(j+1:j+num_f))==0
            loc_time_b=latency(j+1);
            break;
        end
    end
    for j=1:length(label_tag)-num_f
        if label_tag(j)==3 && sum(label_tag(j+1:j+num_f))==0
            loc_time_s=latency(j+1);
            break;
        end
    end
    loc_time(i,1)=loc_time_b-s1;
    loc_time(i,2)=loc_time_s-s2;
    % find ce
    t1=loc_time_b-s1;
    t2=loc_time_s-s1;
    
    temp=time_ce-t1;
    min_idx=min(temp(temp>0));
    idx_t1=find(temp==min_idx);
    
    loc_ce(i,1)=ce(idx_t1);
    
    temp=time_ce-t2;
    min_idx=min(temp(temp>0));
    idx_t2=find(temp==min_idx);
    
    loc_ce(i,2)=ce(idx_t2);
end
save(['ce_at_LOC_',type],'loc_ce');
%% plot
figure;
% dosage={[1:4,6:10],[11:17,19:20],21:30};%MDZ
dosage={1:10,12:20,[21,24:30]};%PPF
range={[0.5 4.5],[0.5 4.5],[0.5 4.5]};%PPF
% range={[0 0.15],[0 0.1],[0 0.1]};%MDZ
for i=1:3
    subplot(1,3,i);
    temp=loc_ce(dosage{i},:);
    [v,idx]=sort(temp(:,1));
    xlabel('Subject Sorted by Onset Time at Baseline');
    ylabel('LOC Time at Scaling');
    temp_=temp(idx,2);
    bar(temp_);
end
dos_tag={'H','M','L'};
figure;
for i=1:3
    subplot(1,3,i);
    temp=loc_ce(dosage{i},:);
    [v,idx]=sort(temp(:,1));
    temp_1=temp(idx,1);
    temp_2=temp(idx,2);
    scatter(temp_1,temp_2,'Diamond');
    xlim(range{i});
    ylim(range{i});
    [rho,pval]=corrcoef(temp_1,temp_2);
    fprintf('[%s] %f %f\n',dos_tag{i},rho(1,2),pval(1,2));
    xlabel('CE at LOC in Non-scaling');
    ylabel('CE at LOC in Scaling');
end    
