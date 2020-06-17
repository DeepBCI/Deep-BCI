clc; clear all; close all;

folder = {'191129_lys','191129_pjy','191202_ljy','191202_ssh','191203_cjh','191206_pjh','191209_chr','191209_ykh','191210_JJH','191211_KGB','191213_MIG','191218_LCB','191219_lht','191219_yyd','191220_ydj'};
% for sub=[1:5,7:15]
for sub=1:15
    close all;
    cd(['D:\NEC lab\Ear-Biosignal\Data_biosignal\' folder{sub}]')
        
    for s=1:3
        rawdata = importdata([folder{sub} '_SSVEP' num2str(s) '.txt'], '\t', 2);
        data(:,:,s) = rawdata.data(:, 1:end-1);
    end
    
    n = 3; Fs = 250; Fn = Fs/2;
    
    [b,a]=butter(n, [59 61]/Fn, 'stop');
    stop_data = filtfilt(b,a,data);
    [b,a]=butter(n, [0.5 100]/Fn, 'bandpass');
    bp = filtfilt(b,a,stop_data);    
    
    for ch=1:9
        Rear_ch{ch} = bp(1:3750,10+ch,:);
        Lear_ch{ch} = bp(1:3750,ch,:);
    end
    for ch=1:3
        Scalp_ch{ch} = bp(1:3750,ch+29,:);
    end       
    
    s1 = 1251; s2 = 3750;
    
    spect = 5:1:15;
    win = 250;
    overlap = 125;
    
    for j = 1:3
        for ch=1:9
            [~, ~, ~, pR_ch{sub,ch}(:,:,j)] = spectrogram(Rear_ch{ch}(s1:s2,:,j), win, overlap, spect, Fs);
            [~, ~, ~, pL_ch{sub,ch}(:,:,j)] = spectrogram(Lear_ch{ch}(s1:s2,:,j), win, overlap, spect, Fs);
        end
        for ch=1:3
            [s, f, t, pS_ch{sub,ch}(:,:,j)] = spectrogram(Scalp_ch{ch}(s1:s2,:,j), win, overlap, spect, Fs);
        end
    end
    
    for ch=1:9
        pR_avr{ch}(:,sub) = mean(mean(pR_ch{sub,ch}(:,:,1:3),3),2);
        pL_avr{ch}(:,sub) = mean(mean(pL_ch{sub,ch}(:,:,1:3),3),2);
        
        SNR(sub,ch) = pR_avr{ch}(6,sub)/mean(pR_avr{ch}([1:5,7:11],sub),1);
        SNR(sub,9+ch) = pL_avr{ch}(6,sub)/mean(pL_avr{ch}([1:5,7:11],sub),1);
    end    
    for ch=1:3
        pS_avr{ch}(:,sub) = mean(mean(pS_ch{sub,ch}(:,:,1:3),3),2);
        SNR(sub,ch+18) = pS_avr{ch}(6,sub)/mean(pS_avr{ch}([1:5,7:11],sub),1);
    end
end

for ch=1:9
    pR_gavr{ch} = mean(pR_avr{ch},2);    pL_gavr{ch} = mean(pL_avr{ch},2);
    pR_err{ch} = std(pR_avr{ch},0,2)/sqrt(length(pR_avr{ch})); pL_err{ch} = std(pL_avr{ch},0,2)/sqrt(length(pR_avr{ch}));
    SNR_avr2(1,ch) = pR_gavr{ch}(6,1)/mean(pR_gavr{ch}([1:5,7:11],1),1);
    SNR_avr2(1,ch+9) = pL_gavr{ch}(6,1)/mean(pL_gavr{ch}([1:5,7:11],1),1);
end

for ch=1:3
    pS_gavr{ch} = mean(pS_avr{ch},2); pS_err{ch} = std(pS_avr{ch},0,2)/sqrt(length(pS_avr{ch}));
    SNR_avr2(1,ch+18) = pS_gavr{ch}(6,1)/mean(pS_gavr{ch}([1:5,7:11],1),1);
    
end

SNR_avr = mean(SNR,1);
% SNR_std = std(SNR);
SNR_err = std(SNR)/sqrt(15);

%% SNR 1-1
%sub7 y2=10 / sub10 y=1.2 / sub11 y=1 / sub12 y=2.2 / sub13 y=1 / sub15 y=1
xscale = [5 20];
yscale = [0 1.2];
yscale2 = [0 3.5];

Rspnum = [8 18 27 36 44 34 25 16 17];
Lspnum = [2 10 19 28 38 30 21 12 11];
Sspnum = [40 41 42];
Sname = {'O1','Oz','O2'};

figure(1)
for ch=1:9
    subplot(5,9,Rspnum(ch));
%     errorbar(10,pR_gavr{ch}(6,1),NaN,pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pR_gavr{ch}(6,1),pR_err{ch}(6,1),pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pR_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on
%     line([10 10],[0 pR_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch),NaN,SNR_err(1,ch),'k')
    errorbar(8.3,SNR_avr(1,ch),SNR_err(1,ch),SNR_err(1,ch),'k')
    text(7.5,SNR_avr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_avr(1,ch),2)),'FontWeight','bold','FontSize',8)
    
    subplot(5,9,Lspnum(ch));
%     errorbar(10,pL_gavr{ch}(6,1),NaN,pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pL_gavr{ch}(6,1),pL_err{ch}(6,1),pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pL_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on
%     line([10 10],[0 pL_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch+9),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch+9),NaN,SNR_err(1,ch+9),'k')
    errorbar(8.3,SNR_avr(1,ch+9),SNR_err(1,ch+9),SNR_err(1,ch+9),'k')
    text(7.5,SNR_avr(1,ch+9)+SNR_err(1,ch+9)+0.5,num2str(round(SNR_avr(1,ch+9),2)),'FontWeight','bold','FontSize',8)
end

for ch=1:3
    subplot(5,9,Sspnum(ch));
%     errorbar(10,pS_gavr{ch}(6,1),NaN,pS_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pS_gavr{ch}(6,1),pS_err{ch}(6,1),pS_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pS_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale2); grid on
%     line([10 10],[0 pS_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(Sname{ch},'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch+18),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch+18),NaN,SNR_err(1,ch+18),'k')
    errorbar(8.3,SNR_avr(1,ch+18),SNR_err(1,ch+18),SNR_err(1,ch+18),'k')
    text(7.5,SNR_avr(1,ch+18)+SNR_err(1,ch+18)+0.5,num2str(round(SNR_avr(1,ch+18),2)),'FontWeight','bold','FontSize',8)
end
supt = suptitle('SSVEP : No re-reference - filtfilt');
set(supt,'FontSize',18,'FontWeight','bold','Color','k')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

%%
% cd('D:\NEC lab\Ear-Biosignal\Figure3\SSVEP_all\SSVEP_no')
% figure(1),saveas (gcf, 'Gavr_SSVEP_ver1-1.jpg')

%% SNR 1-2
xscale = [5 20];

Rspnum = [8 18 27 36 44 34 25 16 17];
Lspnum = [2 10 19 28 38 30 21 12 11];
Sspnum = [40 41 42];
Sname = {'O1','Oz','O2'};

figure(2)
for ch=1:9
    yscale = [0 ceil((pR_gavr{ch}(6,1)+pR_err{ch}(6,1)+0.1)*10^1)/10^1];
    subplot(5,9,Rspnum(ch));
%     errorbar(10,pR_gavr{ch}(6,1),NaN,pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pR_gavr{ch}(6,1),pR_err{ch}(6,1),pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pR_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on   
%     line([10 10],[0 pR_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch),NaN,SNR_err(1,ch),'k')
    errorbar(8.3,SNR_avr(1,ch),SNR_err(1,ch),SNR_err(1,ch),'k')
    text(7.5,SNR_avr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_avr(1,ch),2)),'FontWeight','bold','FontSize',8)
    
    yscale = [0 ceil((pL_gavr{ch}(6,1)+pL_err{ch}(6,1)+0.15)*10^1)/10^1];
    subplot(5,9,Lspnum(ch));
%     errorbar(10,pL_gavr{ch}(6,1),NaN,pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pL_gavr{ch}(6,1),pL_err{ch}(6,1),pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pL_gavr{ch}, 'LineWidth', 2);
    xlim(xscale);ylim(yscale); grid on
%     line([10 10],[0 pL_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch+9),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch+9),NaN,SNR_err(1,ch+9),'k')
    errorbar(8.3,SNR_avr(1,ch+9),SNR_err(1,ch+9),SNR_err(1,ch+9),'k')
    text(7.5,SNR_avr(1,ch+9)+SNR_err(1,ch+9)+0.5,num2str(round(SNR_avr(1,ch+9),2)),'FontWeight','bold','FontSize',8)
end

for ch=1:3
    yscale2 = [0 ceil((pS_gavr{ch}(6,1)+pS_err{ch}(6,1)+0.1)*10^1)/10^1];
    subplot(5,9,Sspnum(ch));    
%     errorbar(10,pS_gavr{ch}(6,1),NaN,pS_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    errorbar(10,pS_gavr{ch}(6,1),pS_err{ch}(6,1),pS_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pS_gavr{ch}, 'LineWidth', 2);
    xlim(xscale);ylim(yscale2); grid on
%     line([10 10],[0 pS_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([15 15],[0 10],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [5 10 15 17.5]; ax1.XTickLabel = {'5', '10', '15', 'SNR'};
    set(gca, 'FontSize', 8,'FontWeight','bold', 'YColor','k',  'XColor','k')
    title(Sname{ch},'FontWeight','bold','FontSize',12);
    ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off'; ax2.FontWeight = 'bold'; ax2.YColor = 'k'; ax2.FontSize = 8;
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 4]; ax2.YTick = [0 2 4]; hold on;
    bar(8.3,SNR_avr(1,ch+18),1,'FaceColor',[0.9290 0.6940 0.1250]);
%     errorbar(8.3,SNR_avr(1,ch+18),NaN,SNR_err(1,ch+18),'k')
    errorbar(8.3,SNR_avr(1,ch+18),SNR_err(1,ch+18),SNR_err(1,ch+18),'k')
    text(7.5,SNR_avr(1,ch+18)+SNR_err(1,ch+18)+0.5,num2str(round(SNR_avr(1,ch+18),2)),'FontWeight','bold','FontSize',8)
end
supt = suptitle('SSVEP : No re-reference - filtfilt');
set(supt,'FontSize',18,'FontWeight','bold','Color','k')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
%%
cd('D:\NEC lab\Ear-Biosignal\Figure\Figure4\SSVEP_all_Gavr')
figure(1),saveas (gcf, 'SSVEP_no_com.jpg')
figure(2),saveas (gcf, 'SSVEP_no_ind.jpg')
