clc; clear all; close all;

folder = {'191129_lys','191129_pjy','191202_ljy','191202_ssh','191203_cjh','191206_pjh','191209_chr','191209_ykh','191210_JJH','191211_KGB','191213_MIG','191218_LCB','191219_lht','191219_yyd','191220_ydj'};
for sub=1:15
    % sub=1;
    % clearvars -except folder sub SNR pp SNR2
    close all;
    
    task = 'Chin';
    cd(['D:\NEC lab\Ear-Biosignal\Data_biosignal\' folder{sub}]')
    
    rawdata = importdata([folder{sub} '_' task '.txt'], '\t', 2);
    
    % start tick
    fid = fopen([folder{sub} '_' task '.txt']);
    start_time = textscan(fid, '%s');
    start = cell2mat(start_time{1,1}(end-12,1));
    start = str2double(start);
    
    n = 3; Fs = 250; Fn = Fs/2;
    
    [b,a]=butter(n, [59 61]/Fn, 'stop');
    stop_data1 = filtfilt(b,a,rawdata.data);
    [b,a]=butter(n, [119 121]/Fn, 'stop');
    stop_data2 = filtfilt(b,a,stop_data1);
    [b,a]=butter(n, [1 124]/Fn, 'bandpass');
    bp = filtfilt(b,a,stop_data2);
    
    % re-reference
    for ch=1:8
        Rear_ch{ch}(:,sub) = bp(start+250:start+15249,10+ch,:)-bp(start+250:start+15249,9,:);
        Lear_ch{ch}(:,sub) = bp(start+250:start+15249,ch,:)-bp(start+250:start+15249,19,:);
    end
    Chin(:,sub) = bp(start+250:start+15249,36);
    
    % RMS
    for j=1:10
        for ch=1:8
            task_R{ch}(:,j) = Rear_ch{ch}(1001+(j-1)*1500:1500+(j-1)*1500,sub);
            
            task_L{ch}(:,j) = Lear_ch{ch}(1001+(j-1)*1500:1500+(j-1)*1500,sub);
        end
        task_Chin(:,j) = Chin(1001+(j-1)*1500:1500+(j-1)*1500,sub);
    end
    for j=1:10
        for ch=1:8
            rest_R{ch}(:,j) = Rear_ch{ch}(251+(j-1)*1500:750+(j-1)*1500,sub);
            
            rest_L{ch}(:,j) = Lear_ch{ch}(251+(j-1)*1500:750+(j-1)*1500,sub);
        end
        rest_Chin(:,j) = Chin(251+(j-1)*1500:750+(j-1)*1500,sub);
    end
    
    for ch=1:8
        rms_task(sub,ch) = mean(rms(task_R{ch}));
        rms_task(sub,ch+8) = mean(rms(task_L{ch}));
        rms_rest(sub,ch) = mean(rms(rest_R{ch}));
        rms_rest(sub,ch+8) = mean(rms(rest_L{ch}));
    end
    rms_task(sub,17) = mean(rms(task_Chin));
    rms_rest(sub,17) = mean(rms(rest_Chin));
    
    % SNR
    for ch=1:17
        SNR(sub,ch) = rms_task(sub,ch)/rms_rest(sub,ch);
    end
end

for ch=1:8
    Rear_gavr{ch} = mean(Rear_ch{ch},2);
    Lear_gavr{ch} = mean(Lear_ch{ch},2);
end
Chin_gavr = mean(Chin,2);

rms_gavr = mean(rms_task);
rms_err = std(rms_task)/sqrt(15);

SNR_gavr = mean(SNR);
SNR_err = std(SNR)/sqrt(15);

%% SNR
% close

% label1 = {'¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','',''};
label1 = {'¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','', 'SNR'};

yrange = [-50 50];
xrange = [0 19000];
xtic = [751 2251 3751 5251 6751 8251 9751 11251 12751 14251 15000];
% xtic = [751:750:15000];
% xtic = [751:1500:15000];
yt = [-50 0 50];

Rspnum = [6 14 21 28 34 26 19 12 13];
Lspnum = [2 8 15 22 30 24 17 10 9];

figure(1)
for ch=1:8
    ax = subplot(5,7,Rspnum(ch)); plot(Rear_gavr{ch});
    yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
    line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
    grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
    set(gca, 'Xtick', xtic);
    set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
    ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 8]; ax2.YTick = [0 4 8]; hold on;
    ax2.XTick = 17000; ax2.XTickLabel = 'SNR';
    bar(17000,SNR_gavr(1,ch),1500,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(17000,SNR_gavr(1,ch),NaN,SNR_err(1,ch),'k')
    text(15800,SNR_gavr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_gavr(1,ch),2)),'FontWeight','bold','FontSize',8)
    
    ax = subplot(5,7,Lspnum(ch)); plot(Lear_gavr{ch});
    yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
    line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
    grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
    set(gca, 'Xtick', xtic);
    set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
    ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 8]; ax2.YTick = [0 4 8]; hold on;
    ax2.XTick = 17000; ax2.XTickLabel = 'SNR';
    bar(17000,SNR_gavr(1,ch+8),1500,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(17000,SNR_gavr(1,ch+8),NaN,SNR_err(1,ch+8),'k')
    text(15800,SNR_gavr(1,ch+8)+SNR_err(1,ch+8)+0.5,num2str(round(SNR_gavr(1,ch+8),2)),'FontWeight','bold','FontSize',8)
end

ax = subplot(5,7,18); plot(Chin_gavr);
yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
set(gca, 'Xtick', xtic);
set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
title('Chin','FontWeight','bold','FontSize',12);
ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 8]; ax2.YTick = [0 4 8]; hold on;
ax2.XTick = 17000; ax2.XTickLabel = 'SNR';
bar(17000,SNR_gavr(1,17),1500,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(17000,SNR_gavr(1,17),NaN,SNR_err(1,17),'k')
text(15800,SNR_gavr(1,17)+SNR_err(1,17)+0.5,num2str(round(SNR_gavr(1,17),2)),'FontWeight','bold','FontSize',8)
supt = suptitle('Chin : Contralateral');
set(supt,'FontSize',18,'FontWeight','bold','Color','k')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

%%
cd('D:\NEC lab\Ear-Biosignal\Figure\Figure4\Chin_Gavr_2s')
figure(1),saveas (gcf, 'Gavr_cont_SNR.jpg')

%% RMS
% close

% label1 = {'¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','','¡è','',''};
label1 = {'¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','¡è','', 'RMS'};

yrange = [-50 50];
xrange = [0 19000];
xtic = [751 2251 3751 5251 6751 8251 9751 11251 12751 14251 15000];
% xtic = [751:750:15000];
% xtic = [751:1500:15000];
yt = [-50 0 50];

Rspnum = [6 14 21 28 34 26 19 12 13];
Lspnum = [2 8 15 22 30 24 17 10 9];

figure(2)
for ch=1:8
    ax = subplot(5,7,Rspnum(ch)); plot(Rear_gavr{ch});
    yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
    line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
    grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
    set(gca, 'Xtick', xtic);
    set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
    ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 80]; ax2.YTick = [0 40 80]; hold on;
    ax2.XTick = 17000; ax2.XTickLabel = 'RMS';
    bar(17000,rms_gavr(1,ch),1500,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(17000,rms_gavr(1,ch),NaN,rms_err(1,ch),'k')
    text(15500,rms_gavr(1,ch)+rms_err(1,ch)+5,num2str(round(rms_gavr(1,ch),2)),'FontWeight','bold','FontSize',8)
    
    ax = subplot(5,7,Lspnum(ch)); plot(Lear_gavr{ch});
    yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
    line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
    grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
    set(gca, 'Xtick', xtic);
    set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
    ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 80]; ax2.YTick = [0 40 80]; hold on;
    ax2.XTick = 17000; ax2.XTickLabel = 'RMS';
    bar(17000,rms_gavr(1,ch+8),1500,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(17000,rms_gavr(1,ch+8),NaN,rms_err(1,ch+8),'k')
    text(15500,rms_gavr(1,ch+8)+rms_err(1,ch+8)+5,num2str(round(rms_gavr(1,ch+8),2)),'FontWeight','bold','FontSize',8)
end

ax = subplot(5,7,18); plot(Chin_gavr);
yticks(yt); ylim(yrange); xlim(xrange); ax.XColor = 'k'; ax.YColor = 'k';
line([15000 15000],[-500 500],'LineWidth',0.5, 'Color', [0 0 0]);
grid on; ax.GridColor = 'k'; ax.GridAlpha = 0.8; ax.YGrid = 'off';
set(gca, 'Xtick', xtic);
set(gca, 'XtickLabel', label1, 'FontSize', 10,'FontWeight','bold')
title('Chin','FontWeight','bold','FontSize',12);
ax2 = axes('Position',ax.Position, 'XAxisLocation','bottom', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'on'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 10;
ax2.XAxis.Limits = [0 19000]; ax2.YAxis.Limits = [0 80]; ax2.YTick = [0 40 80]; hold on;
ax2.XTick = 17000; ax2.XTickLabel = 'RMS';
bar(17000,rms_gavr(1,17),1500,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(17000,rms_gavr(1,17),NaN,rms_err(1,17),'k')
text(15500,rms_gavr(1,17)+rms_err(1,17)+5,num2str(round(rms_gavr(1,17),2)),'FontWeight','bold','FontSize',8)
supt = suptitle('Chin : Contralateral');
set(supt,'FontSize',18,'FontWeight','bold','Color','k')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

%%
cd('D:\NEC lab\Ear-Biosignal\Figure\Figure4\Chin_Gavr_2s')
figure(2),saveas (gcf, 'Gavr_cont_RMS.jpg')
