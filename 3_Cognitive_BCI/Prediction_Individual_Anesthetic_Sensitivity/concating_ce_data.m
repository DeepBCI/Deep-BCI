% concating_ce_data.m
%
% Since there are differences of time point between ce and label latency,
% this code matched the time point.
%
% author: Young-Seok Kweon
% created: 2020.02.07
%% init
clc;clear;close all;
%% load
type={'PPF','MDZ'};
dosage={'H','M','L'};
n=10;%number of subjects in each group
for a=2
for d=1:3
    for i=1:n
        x=load(['with_scale\',dosage{d},type{a}(1),'_S',num2str(i)]);
        y=load(['E:\Code\ConvertedData\',dosage{d},type{a}(1),'_S',num2str(i),'_bio']);
        
        data=x.data;
        event=x.event;
        raw_event=x.raw_event;
        time_ce=y.time_cpce;
        ce=y.ce;
        
        % event에 해당하는 latency 구하기
        if (d==2 && i==1) && a==1
            raw_event=hj_transfer(raw_event);
        end
        event=struct2cell(event);
        mrk_data=cell2mat(event(8,:,:));
        mrk_data=reshape(mrk_data,size(mrk_data,3),1);
        
        raw_event=struct2cell(raw_event);
        trigger=raw_event(6,:,:);
        mrk_orig=cell2mat(raw_event(8,:,:));
        mrk_orig=reshape(mrk_orig,size(mrk_orig,3),1);
        latency=cell2mat(raw_event(1,:,:));
        latency=reshape(latency,size(latency,3),1);
        idx=logical(sum(mrk_orig==mrk_data',2));
        latency_data=latency(idx); %event에 해당하는 latency
        trigger_data=trigger(idx);
        % 쓰잘데기 없는 trigger 제거
        idx=[];cnt=0;
        for k=1:length(trigger_data)
            if ~(length(trigger_data{k})==4)
                cnt=cnt+1;
                idx(cnt)=k;
            end
        end
        trigger_data(idx)=[];latency_data(idx)=[];
        trigger_data=cell2mat(trigger_data); % 'S  2'라 1x4xevent수임
        trigger_data=str2num(reshape(trigger_data(:,4,:),size(trigger_data(:,4,:),3),1)); % '2'->2
        
        % event중 첫번째 'S  3' 이후에 나오는 'S  8'이 0 sec
        latency_data=latency_data./1000;% 1000Hz임으로 sec로 변환
        for temp_idx=2:length(trigger_data)
            if trigger_data(temp_idx)==8 && trigger_data(temp_idx-1)==3
                break;
            end
        end        
        start_idx_b=temp_idx;
        for temp_idx=2:length(trigger_data)
            if trigger_data(temp_idx)==8 && trigger_data(temp_idx-1)==6
                break;
            end
        end
        start_idx_s=temp_idx;
        

        % event중에 auditory stimulation만 추출 (num_event>num_trial_data -> 동일하게 맞춰줌)
        cnt=0;label=[];label_idx=[];
        for temp_idx=1:length(trigger_data)-1
            if trigger_data(temp_idx)==2 && trigger_data(temp_idx+1)==8
                cnt=cnt+1;
                label(cnt)=2;
                label_idx(cnt)=temp_idx;
            elseif trigger_data(temp_idx)==3 && trigger_data(temp_idx+1)==8
                cnt=cnt+1;
                label(cnt)=1;
                label_idx(cnt)=temp_idx;
            elseif trigger_data(temp_idx)==6 && trigger_data(temp_idx+1)==8
                cnt=cnt+1;
                label(cnt)=3;
                label_idx(cnt)=temp_idx;
            elseif (trigger_data(temp_idx)==2 || trigger_data(temp_idx)==3 || trigger_data(temp_idx)==6) && ~(trigger_data(temp_idx+1)==8)
                cnt=cnt+1;
                label(cnt)=0;
                label_idx(cnt)=temp_idx;
            end
        end
        latency_label=latency_data(label_idx);
        % jsWoo 피실험자는 baseline ('S  3') 시작하고 바로 PPF 주입된게 아닌거 같음
        % 시작하자마자 바로 눌렀다고 생각하면 누른시점이랑 주입시점이랑 매치가 안됨
        % [d=3,i=2]
        
        if d==3 && i==2 && a==1
            infusion_time={'11:14:53','11:16:12','11:17:29','11:18:47','11:23:58','11:25:14'...
                '11:34:01','11:35:01','11:42:44','11:55:31','11:56:47','11:59:30',...
                '12:04:25','12:05:35','12:11:34','12:14:33','12:19:13'}';
            infusion_time=cell2mat(infusion_time);
            itime.hr=str2num(infusion_time(:,1:2)); %#ok<ST2NM>
            itime.min=str2num(infusion_time(:,4:5)); %#ok<ST2NM>
            itime.sec=str2num(infusion_time(:,7:8)); %#ok<ST2NM>
            m=length(infusion_time);
            for temp_idx=1:m-1
                interv(temp_idx)=(itime.hr(temp_idx+1)-itime.hr(temp_idx))*3600+...
                    (itime.min(temp_idx+1)-itime.min(temp_idx))*60+...
                    (itime.sec(temp_idx+1)-itime.sec(temp_idx));
            end
            interv_accum(1)=interv(1);
            for temp_idx=2:m-1
                interv_accum(temp_idx)=interv_accum(temp_idx-1)+interv(temp_idx);
            end
            for temp_idx=1:200
                temp=latency_data(start_idx_b+temp_idx-1);
                interv_new(1)=temp;
                interv_new(2:m)=interv_accum+temp;
                loss(temp_idx)=0;
                for j=1:m
                    temp=latency_label-interv_new(j);
                    min_t=min(temp(temp>0));
                    idx=find(min_t==temp);
                    loss(temp_idx)=loss(temp_idx)+(label(idx)==0);
                end
            end
            [M,I]=min(loss);
            start_idx_b=start_idx_b+I;
        end
        
        time_0base=latency_data(start_idx_b);
        latency_data=latency_data-time_0base;
        time_0scale=latency_data(start_idx_s);
        latency_label=latency_data(label_idx); 
        
        % ce 뒤에 너무 길게 남아서 쳐냄
        temp_t=latency_label(end);
        temp=time_ce-temp_t;
        min_t=min(temp(temp>0));
        if isempty(min_t)
            idx=length(time_ce);
        else
            idx=find(min_t==temp);
        end
        logic=logical(label);
        
        % plot
        figure;
        yyaxis left;
        stem(latency_label,label,'Marker','.','MarkerSize',3);
        hold on;
        yyaxis right;
        plot(time_ce(1:idx),ce(1:idx),'LineWidth',2);
        
        % save
        savefig(['ce\',type{a},'_S',num2str((d-1)*10+i)]);
        close all;
        save(['ce\',type{a},'_S',num2str((d-1)*10+i)],'data','latency_label','label','time_ce','ce','-v7.3');
        fprintf('[S%d]:done\n',(d-1)*10+i);
    end
end
end

%% hj marker change
function y=hj_transfer(x)
n=size(x,2);
cnt=0;session=0;
for i=1:n
    if strcmp(x(i).type,'S 11')==1 && session==0
        session=2;
    elseif strcmp(x(i).type,'S 11')==1 && session==2
        session=3;
    elseif strcmp(x(i).type,'S 12')==1 && session==3
        session=6;
    end
    if strcmp(x(i).type,'S  1')==1
        x(i).type=['S  ',num2str(session)];
        cnt=cnt+1;
    elseif strcmp(x(i).type,'S  1')==1
        x(i).type=['S  ',num2str(session)];
        cnt=cnt+1;
    elseif strcmp(x(i).type,'S  2')==1
        x(i).type=['S  ',num2str(session)];
        cnt=cnt+1;
    end
end
y=x;
end
