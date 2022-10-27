% PTE_2_plot.m
% Phase transfer entropy 
%
% created: 2022.10.11
% author: Gi-Hwan Shin
%% init
clc; clear; close all;
%% path setting
temp = pwd;
list = split(temp,'\');

path = [];
for i=1:length(list)-2
    path = [path,list{i},'\'];
end
%% addpath
addpath([path,'Wake_Sleep\Lib\eeglab14_1_2b\']);
eeglab;

%% load data
range = {[1 3.5],[4 7.5],[8 13.5],[14 29.5],[30 50]}; % Delta, Theta, Alpha, Beta, Gamma

removal = [9,11,21,24,26,33,35,36]; % non resting-state

region = {[1:4 26:33 56:59], [5:7 21:22 24:25 35:37 52:54],...
    [8 34 38],[23 51 55], [9:13 17:20 39:41 47:50], [14:16 42:46 60]}; % F,C,LT,RT,P,O
RE = cat(2,region{:});

% mycolormap = customcolormap(linspace(0,1,11), {'#68011d','#b5172f','#d75f4e','#f7a580','#fedbc9','#f5f9f3','#d5e2f0','#93c5dc','#4295c1','#2265ad','#062e61'});
mycolormap = customcolormap([0 .25 .5 .75 1], {'#9d0142','#f66e45','#ffffbb','#65c0ae','#5e4f9f'});

% mycolormap = redblue;
cnt=0;
for n=1:37
    if sum(n == removal) == 1
        continue;
    end
    cnt=cnt+1;
    
    % dPTE / PTE (chxchxmethodssxrange)
    for t=1:3
        load([path 'Winter_2023\Analysis\2_PTE_lap\sub' num2str(n) '_PTE_' num2str(t)]); %chxchxmethodxrange
        Data_s(:,:,cnt,:,t)=dPTE(:,:,1,:); %chxchxsubxrangextime
        Data_o(:,:,cnt,:,t)=dPTE(:,:,2,:); %chxchxsubxrangextime
    end
end

%% dPTE
% norm(-0.5)
Data_sco=Data_s-0.5;
Data_ont=Data_o-0.5;

% Diagonal zeros
for t=1:3
    for r=1:size(range,2)
        for n=1:cnt
            % channels
            D_S(:,:,n,r,t)=Data_sco(:,:,n,r,t) - diag(diag(Data_sco(:,:,n,r,t))); %chxchxsubxrangextime
            D_O(:,:,n,r,t)=Data_ont(:,:,n,r,t) - diag(diag(Data_ont(:,:,n,r,t)));
            
            % ROIs
            for re1=1:size(region,2)
                for re2=1:size(region,2)
                    S_ROIs(re1,re2,n,r,t)=mean(nonzeros(D_S(region{re1},region{re2},n,r,t)),1);
                    O_ROIs(re1,re2,n,r,t)=mean(nonzeros(D_O(region{re1},region{re2},n,r,t)),1);
                end
            end
            S_ROI(:,:,n,r,t) = S_ROIs(:,:,n,r,t)-S_ROIs(:,:,n,r,t)';  % rexrexsubxrangextime
            O_ROI(:,:,n,r,t) = O_ROIs(:,:,n,r,t)-O_ROIs(:,:,n,r,t)'; 
        end
    end
end

% channels
D_S=D_S(RE,RE,:,:,:);
D_O=D_O(RE,RE,:,:,:);

%% dPTE Plot
% avg subjects
S_avg=squeeze(mean(D_S,3)); %chxchxrangextime
O_avg=squeeze(mean(D_O,3)); 

% scott
cnt=0;
figure
for r=1:size(range,2)
    max_t = max(abs(S_avg(:,:,r,:)),[],'all');
    for t=1:3
        cnt=cnt+1;
        subplot(length(range),3,cnt); 
             
        image(S_avg(:,:,r,t),'CDataMapping','scaled');
    
        colormap(mycolormap)
        colorbar
        caxis([-max_t max_t]);

        xticks([8,22.5,31,34,44,55.5]);
        xticklabels({'F','C','LT','RT','P','O'});
        yticks([8,22.5,31,34,44,55.5]);
        yticklabels({'F','C','LT','RT','P','O'});      
    end
end

% otnes
cnt=0;
figure
for r=1:size(range,2)
    max_t = max(abs(O_avg(:,:,r,:)),[],'all');
    for t=1:3
%         figure
        cnt=cnt+1;
        subplot(length(range),3,cnt); 
             
        image(O_avg(:,:,r,t),'CDataMapping','scaled');
        
        colormap(mycolormap)
        colorbar
        caxis([-max_t max_t]);

        xticks([8,22.5,31,34,44,55.5]);
        xticklabels({'',' ',' ',' ',' ',' '});
        yticks([8,22.5,31,34,44,55.5]);
        yticklabels({' ',' ',' ',' ',' ',' '});      
    end
end

%% statistical analysis - channels
% D_S=permute(D_S,[3,1,2,4,5]); %subxchxchxrangextime
% D_O=permute(D_O,[3,1,2,4,5]); 
% 
% % among Resting-state EEGs
% cnt=0;
% for t1=1:2
%     for t2=t1+1:3
%         cnt=cnt+1;
%         for r=1:size(range,2)
%             for c1=1:60
%                 for c2=1:60
%                     [h_s,p_s(c1,c2,r,cnt),ci_s,stats_s] = ttest(D_S(:,c1,c2,r,t1),D_S(:,c1,c2,r,t2));
%                     t_s(c1,c2,r,cnt)=stats_s.tstat;
%                     [h_o,p_o(c1,c2,r,cnt),ci_o,stats_o] = ttest(D_O(:,c1,c2,r,t1),D_O(:,c1,c2,r,t2));
%                     t_o(c1,c2,r,cnt)=stats_o.tstat;
%                 end
%             end
%         end
%     end
% end
% 
% % Figure
% % scott
% figure
% cnt = 0;
% for r = 1:size(range,2) % frequency band
%     max_t = max(abs(t_s(:,:,r,:)),[],'all');
%     for t = 1:3
%         cnt = cnt+1;
%         
%         subplot(length(range),3,cnt)
%         temp = t_s(:,:,r,t);
%         temp(p_s(:,:,r,t)>0.01)=0; % ttest
%         temp(isnan(temp))=0; %nan to 1
%         
%         image(temp,'CDataMapping','scaled');
%         colormap(mycolormap)
%         colorbar
%         caxis([-max_t max_t]);
% 
%         xticks([8,22.5,31,34,44,55.5]);
%         xticklabels({'F','C','LT','RT','P','O'});
%         yticks([8,22.5,31,34,44,55.5]);
%         yticklabels({'F','C','LT','RT','P','O'});
%     end
% end
% 
% % otnes
% figure
% cnt = 0;
% for r = 1:size(range,2) % frequency band
%     max_t = max(abs(t_s(:,:,r,:)),[],'all');
%     for t = 1:3
% %         figure
%         cnt = cnt+1;
%         
%         subplot(length(range),3,cnt)
%         temp = t_o(:,:,r,t);
%         temp(p_o(:,:,r,t)>0.01)=0; % ttest
%         temp(isnan(temp))=0; %nan to 1
%         
%         image(temp,'CDataMapping','scaled');
%         colormap(mycolormap)
% %         colorbar
%         caxis([-max_t max_t]);
%         
%         xticks([8,22.5,31,34,44,55.5]);
%         xticklabels({'F','C','LT','RT','P','O'});
%         yticks([8,22.5,31,34,44,55.5]);
%         yticklabels({'F','C','LT','RT','P','O'});
%         
%     end
% end

%% statistical analysis - ROIs
S_ROI = permute(S_ROI,[3 1 2 4 5]); %subxrexrexrangextime
O_ROI = permute(O_ROI,[3 1 2 4 5]);


% among brain regionds
% for t=1:3
%     for r=1:size(range,2)
%         for re1=1:size(region,2)-1
%             for re2=re1+1:size(region,2)
%                 [h_br,p_br(re1,re2,r,t),ci_br,stat_br]=ttest(O_ROI(:,1,1,r,t),O_ROI(:,re1,re2,r,t));
%                 t_br(re1,re2,r,t)=stat_br.tstat;
%             end
%         end
%     end
% end


% among Resting-state EEGs
cnt = 0;
for t1=1:2
    for t2=t1+1:3
       cnt=cnt+1;
       for r=1:size(range,2)
           for re1=1:size(region,2)
               for re2=1:size(region,2)
                   % ttest
%                    [h_s_re,p_s_re(re1,re2,r,cnt),ci_s_re,stat_s_re] = ttest(S_ROI(:,re1,re2,r,t1),S_ROI(:,re1,re2,r,t2));
%                    t_s_re(re1,re2,r,cnt)=stat_s_re.tstat;
%                    [h_o_re,p_o_re(re1,re2,r,cnt),ci_o_re,stat_o_re] = ttest(O_ROI(:,re1,re2,r,t1),O_ROI(:,re1,re2,r,t2));
%                    t_o_re(re1,re2,r,cnt)=stat_o_re.tstat;
                   
                   % permutation
                   [t_s_re(re1,re2,r,cnt),df_s_re,p_s_re(re1,re2,r,cnt)] = statcond({S_ROI(:,re1,re2,r,t1)' S_ROI(:,re1,re2,r,t2)'},'paired','on','method','perm','naccu',5000);
                   [t_o_re(re1,re2,r,cnt),df_o_re,p_o_re(re1,re2,r,cnt)] = statcond({O_ROI(:,re1,re2,r,t1)' O_ROI(:,re1,re2,r,t2)'},'paired','on','method','perm','naccu',5000);
               end
           end
       end
    end
end

% Figure
% scott
figure
cnt = 0;
for r=1:size(range,2)
    max_t = max(abs(t_s_re(:,:,r,:)),[],'all');
    for t=1:3
        cnt = cnt+1;
        
        subplot(length(range),3,cnt)
        temp = t_s_re(:,:,r,t);
        temp(p_s_re(:,:,r,t)>0.01)=0; % ttest
        temp(isnan(temp))=0; %nan to 1
        
        image(temp,'CDataMapping','scaled');
        colormap(mycolormap)
        colorbar
        caxis([-max_t max_t]);

        xticks([1 2 3 4 5 6]);
        xticklabels({'F','C','LT','RT','P','O'});
        yticks([1 2 3 4 5 6]);
        yticklabels({'F','C','LT','RT','P','O'});        
    end
end
% otnes
figure
cnt = 0;
for r=1:size(range,2)
    max_t = max(abs(t_o_re(:,:,r,:)),[],'all');
    for t=1:3
        cnt = cnt+1;
%         figure
        subplot(length(range),3,cnt)
        temp = t_o_re(:,:,r,t);
        temp(p_o_re(:,:,r,t)>0.01)=0; % ttest
        temp(isnan(temp))=0; %nan to 1
        
        image(temp,'CDataMapping','scaled');
        colormap(mycolormap)
        colorbar
        caxis([-max_t max_t]);

        xticks([1 2 3 4 5 6]);
        xticklabels({' ',' ',' ',' ',' ',' '});
        yticks([1 2 3 4 5 6]);
        yticklabels({' ',' ',' ',' ',' ',' '});
        
        % significant EEG (row,col)
        [row, col]=find(triu( temp));
        ROI_row{r,t}=row;
        ROI_col{r,t}=col;
    end
end

%% Correlation
cnt=0;
for r=1:size(range,2)
    for t=1:3
        if ROI_row{r,t} ~= 0
            for i=1:size(ROI_row{r,t},1)
                if t==1 % RS 2 - RS 1
                    cnt=cnt+1;
                    SIG(:,cnt)=O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,2)-O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,1);
                elseif t==2 % RS 3 - RS 1
                    cnt=cnt+1;
                    SIG(:,cnt)=O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,3)-O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,1);
                else % RS 3 - RS 2
                    cnt=cnt+1;
                    SIG(:,cnt)=O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,3)-O_ROI(:,ROI_row{r,t}(i),ROI_col{r,t}(i),r,2);
                end
            end
        end
    end
end

load('ACC.mat') % Acc
Acc(:,29)=[];
for i=1:size(SIG,2)
    figure
    [R_A P_A] = corrcoef(SIG(:,i),Acc);%
    RR_DA(i)=R_A(1,2); 
    PP_DA(i)=P_A(1,2);  

    scatter(SIG(:,i),Acc,50,'filled','k');
    hold on
    scatter_line_R = polyfit(SIG(:,i),Acc', 1);
    set(gca, 'FontSize', 15)
    x1 = linspace(min(SIG(:,i)), max(SIG(:,i)), 1000);
    y1 = polyval(scatter_line_R, x1);
    hold on
    plot(x1,y1,'k--','LineWidth',1);
end
