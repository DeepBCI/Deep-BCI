% PSD_1_plot.m
% Power spectral density
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

%%
range = {[1 3.5],[4 7.5],[8 13.5],[14 29.5],[30 50]}; % Delta, Theta, Alpha, Beta, Gamma
removal = [9,11,21,24,26,33,35,36]; % non resting-state

region = {[1:4 26:33 56:59], [5:7 21:22 24:25 35:37 52:54],...
    [8 34 38],[23 51 55], [9:13 17:20 39:41 47:50], [14:16 42:46 60]}; % F,C,LT,RT,P,O
%% load 
cnt = 0;
for n=1:37
    if sum(n == removal) == 1
        continue;
    end
    cnt=cnt+1;
    
    for t=1:3
        load([path 'Winter_2023\Analysis\1_PSD_lap\sub' num2str(n) '_PSD_' num2str(t)]);
        Data(:,:,t,cnt)=Data_PSD; % chxrangextimexsub
    end
end

%% Statistical analysis
Data = permute(Data,[4 1 2 3]); % subxchxrangextime

% anova by channel
% Data_anova = permute(Data,[1 4 2 3]); % sub x condition x ch x freq
% for r = 1:size(range,2)
%     for c = 1:60
%         anova_p(c,r) = anova1(Data_anova(:,:,c,r),[],'off');
%     end
% end

% post-hoc analysis
cnt = 0;
for t1 = 1:2
    for t2 = t1+1:3
        cnt =cnt+1; 
        for r = 1:size(range,2)
            for c = 1:60
                [h, p(c,r,cnt), ci, stats] = ttest(Data(:,c,r,t1), Data(:,c,r,t2));
                stat_s(c,r,cnt) = stats.tstat;            
            end
        end
    end
end

% Figure
load('EEG.mat');
figure
cnt = 0;
mycolormap = customcolormap(linspace(0,1,11), {'#68011d','#b5172f','#d75f4e','#f7a580','#fedbc9','#f5f9f3','#d5e2f0','#93c5dc','#4295c1','#2265ad','#062e61'});
for t = 1:3
    max_t = max(abs(stat_s(:,:,t)),[],'all');
    for r = 1:size(range,2) % frequency band
        cnt = cnt+1;
        
        subplot(3,length(range),cnt)
        
%         p((anova_p(:,r)>0.05),r,t)=1;
        
        topoplot(stat_s(:,r,t), EEG, 'emarker2',{find(p(:,r,t)<0.01),'*',[0.9,0.9,0.9]}, 'maplimits', [-max_t max_t],'colormap', mycolormap);
      
        colormap(mycolormap)
        colorbar;
    end
end

%% Correlation - ROIs
cnt=0;
for t=1:3
    for r=1:size(range,2)
        sig=find(p(:,r,t)<0.01);
        
        if sig ~= 0
            for re=1:size(region,2) % 6
                Sig_ch=intersect(region{re},sig); % region x range x time
                
                if Sig_ch ~=0
                    if t==1 % RS 2 - RS 1
                        cnt=cnt+1;
                        SIG(:,cnt)=mean(Data(:,Sig_ch,r,2),2)-mean(Data(:,Sig_ch,r,1),2);
                    elseif t==2 % RS 3 - RS 1
                        cnt=cnt+1;
                        SIG(:,cnt)=mean(Data(:,Sig_ch,r,3),2)-mean(Data(:,Sig_ch,r,1),2);
                    else % RS 3 - RS 2
                        cnt=cnt+1;
                        SIG(:,cnt)=mean(Data(:,Sig_ch,r,3),2)-mean(Data(:,Sig_ch,r,2),2);
                    end
                end
            end
        end
    end
end

load('ACC.mat') % Acc
Acc(:,29)=[];

for i=1:size(SIG,2)
    [R_A P_A] = corrcoef(SIG(:,i),Acc);%
    RR_DA(i)=R_A(1,2); 
    PP_DA(i)=P_A(1,2); 

    figure
    scatter(SIG(:,i),Acc,50,'filled','k');
    hold on
    scatter_line_R = polyfit(SIG(:,i),Acc', 1);
    set(gca, 'FontSize', 15)
    x1 = linspace(min(SIG(:,i)), max(SIG(:,i)), 1000);
    y1 = polyval(scatter_line_R, x1);
    hold on
    plot(x1,y1,'k--','LineWidth',1);       
end

%% detection outlier
% Data % subxchxrangextime
% for ch=1:60
%     for r=1:5
%         for t=1:3
%             TF{ch,r,t} = find(isoutlier(Data(:,ch,r,t)));
%         end
%     end
% end
% TF=permute(TF,[1 3 2]);