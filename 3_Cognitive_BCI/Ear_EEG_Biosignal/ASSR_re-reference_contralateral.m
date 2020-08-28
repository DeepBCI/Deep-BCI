clc; clear all; close all;

folder = {'191129_lys','191129_pjy','191202_ljy','191202_ssh','191203_cjh','191206_pjh','191209_chr','191209_ykh','191210_JJH','191211_KGB','191213_MIG','191218_LCB','191219_lht','191219_yyd','191220_ydj'};
% for sub=[1:5,7:15]
for sub=1:15
    if sub==6
        pp=26;
    else
        pp=29;
    end
    clearvars -except folder sub SNR pp pPz pPz_avr pCz pCz_avr pR_ch pR_avr pL_ch pL_avr
    close all;
    
    hertz = '1000';
    
    cd(['D:\NEC lab\Ear-Biosignal\Data_biosignal\' folder{sub}]')
    if sub == 1 || sub == 2
        for i=1:3
            rawdata2{i} = importdata([folder{sub} '_ASSR_' hertz '_40_' num2str(i) '.txt'], '\t', 2);
        end
        
        n = 3; Fs = 250; Fn = Fs/2;
        for k=1:3
            [b,a]=butter(n, [58 62]/Fn, 'stop');
            stop_data = filtfilt(b,a,rawdata2{k}.data);
            [b,a]=butter(n, [0.5 100]/Fn, 'bandpass');
            bp(:,:,k) = filtfilt(b,a,stop_data);
        end
        
        for h=1:3
            Hz(:,:,h) = bp(251:end,:,h);
        end
        Hzs=[Hz ; zeros(250,41,3)];
    else
        rawdata = importdata([folder{sub} '_ASSR_' hertz '.txt'], '\t', 2);
        
        n = 3; Fs = 250; Fn = Fs/2;
        
        [b,a]=butter(n, [59 61]/Fn, 'stop');
        stop_data = filtfilt(b,a,rawdata.data);
        [b,a]=butter(n, [0.5 100]/Fn, 'bandpass');
        bp = filtfilt(b,a,stop_data);
        start = 850;
        
        for h=1:3
            Hzs(:,:,h) = bp(1+start+(h-1)*6250:3750+start+(h-1)*6250,:);
        end
    end
    
    %%
    
    for ch=1:8
        Rear_ch{ch} = Hzs(1:3750,10+ch,:)-Hzs(1:3750,9,:);
        Lear_ch{ch} = Hzs(1:3750,ch,:)-Hzs(1:3750,19,:);
    end
    
    s1 = 1250; s2 = 3750;
    
    spect = 35:1:45;
    
    win = 250;
    overlap = 125;
    
    for j = 1:3
        for ch=1:8
            [~, ~, ~, pR_ch{sub,ch}(:,:,j)] = spectrogram(Rear_ch{ch}(s1:s2,:,j), win, overlap, spect, Fs);
            [~, ~, ~, pL_ch{sub,ch}(:,:,j)] = spectrogram(Lear_ch{ch}(s1:s2,:,j), win, overlap, spect, Fs);
        end
        [~, ~, ~, pCz{sub}(:,:,j)] = spectrogram(Hzs(s1:s2,28,j), win, overlap, spect, Fs);
        [s, f, t, pPz{sub}(:,:,j)] = spectrogram(Hzs(s1:s2,pp,j), win, overlap, spect, Fs);
    end
    
    for ch=1:8
        pR_avr{ch}(:,sub) = mean(mean(pR_ch{sub,ch}(:,:,1:3),3),2);
        pL_avr{ch}(:,sub) = mean(mean(pL_ch{sub,ch}(:,:,1:3),3),2);
        
        SNR(sub,ch) = pR_avr{ch}(6,sub)/mean(pR_avr{ch}([1:5,7:11],sub),1);
        SNR(sub,8+ch) = pL_avr{ch}(6,sub)/mean(pL_avr{ch}([1:5,7:11],sub),1);
    end
    pCz_avr(:,sub) = mean(mean(pCz{sub}(:,:,1:3),3),2); pPz_avr(:,sub) = mean(mean(pPz{sub}(:,:,1:3),3),2);
    SNR(sub,17) = pCz_avr(6,sub)/mean(pCz_avr([1:5,7:11],sub),1); SNR(sub,18) = pPz_avr(6,sub)/mean(pPz_avr([1:5,7:11],sub),1);
end

for ch=1:8
    pR_gavr{ch} = mean(pR_avr{ch},2);    pL_gavr{ch} = mean(pL_avr{ch},2);
    pR_std{ch} = std(pR_avr{ch},0,2); pL_std{ch} = std(pL_avr{ch},0,2);
    pR_err{ch} = std(pR_avr{ch},0,2)/sqrt(length(pR_avr{ch})); pL_err{ch} = std(pL_avr{ch},0,2)/sqrt(length(pR_avr{ch})); 
    SNR_avr2(1,ch) = pR_gavr{ch}(6,1)/mean(pR_gavr{ch}([1:5,7:11],1),1);
    SNR_avr2(1,ch+8) = pL_gavr{ch}(6,1)/mean(pL_gavr{ch}([1:5,7:11],1),1);
end
pCz_gavr = mean(pCz_avr,2);     pPz_gavr = mean(pPz_avr,2);
pCz_std = std(pCz_avr,0,2);     pPz_std = std(pPz_avr,0,2);
pCz_err = std(pCz_avr,0,2)/sqrt(length(pCz_avr));     pPz_err = std(pPz_avr,0,2)/sqrt(length(pPz_avr)); 

SNR_avr = mean(SNR,1);
% SNR_std = std(SNR);
SNR_err = std(SNR)/sqrt(15);
% SNR_avr2(1,17) = pCz_gavr(6,1)/mean(pCz_gavr([1:5,7:11],1),1);
% SNR_avr2(1,18) = pPz_gavr(6,1)/mean(pPz_gavr([1:5,7:11],1),1);

%% SNR 1-1

spect = 35:1:45;
xscale = [35 50];
yscale = [0 1.5]; %[0 40];
yscale2 = [0 1.5];

Rspnum = [8 18 27 36 44 34 25 16 17];
Lspnum = [2 10 19 28 38 30 21 12 11];

figure(1)
for ch=1:8
    subplot(5,9,Rspnum(ch));
    errorbar(40,pR_gavr{ch}(6,1),NaN,pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pR_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on
    line([40 40],[0 pR_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([45 45],[0 10],'LineWidth',0.5, 'Color', 'k')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
    bar(8.3,SNR_avr(1,ch),1,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(8.3,SNR_avr(1,ch),NaN,SNR_err(1,ch),'k')
    text(7.5,SNR_avr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_avr(1,ch),2)))
    
    subplot(5,9,Lspnum(ch));
    errorbar(40,pL_gavr{ch}(6,1),NaN,pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pL_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on
    line([40 40],[0 pL_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([45 45],[0 10],'LineWidth',0.5, 'Color', 'k')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
    bar(8.3,SNR_avr(1,ch+8),1,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(8.3,SNR_avr(1,ch+8),NaN,SNR_err(1,ch+8),'k')
    text(7.5,SNR_avr(1,ch+8)+SNR_err(1,ch+8)+0.5,num2str(round(SNR_avr(1,ch+8),2)))
end
subplot(5,9,23);
errorbar(40,pCz_gavr(6,1),NaN,pCz_err(6,1),'Color',[0 0.4470 0.7410]); hold on;
plot(f,pCz_gavr, 'LineWidth', 2);
xlim(xscale); ylim(yscale2); grid on
line([40 40],[0 pCz_gavr(6,1)],'LineWidth',1, 'Color', 'r')
line([45 45],[0 10],'LineWidth',0.5, 'Color', 'k')
title('Cz','FontWeight','bold','FontSize',12);
ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
bar(8.3,SNR_avr(1,17),1,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(8.3,SNR_avr(1,17),NaN,SNR_err(1,17),'k')
text(7.5,SNR_avr(1,17)+SNR_err(1,17)+0.5,num2str(round(SNR_avr(1,17),2)))

subplot(5,9,32);
errorbar(40,pPz_gavr(6,1),NaN,pPz_err(6,1),'Color',[0 0.4470 0.7410]); hold on;
plot(f,pPz_gavr, 'LineWidth', 2);
xlim(xscale); ylim(yscale2); grid on
line([40 40],[0 pPz_gavr(6,1)],'LineWidth',1, 'Color', 'r')
line([45 45],[0 10],'LineWidth',0.5, 'Color', 'k')
title('Pz','FontWeight','bold','FontSize',12);
ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
bar(8.3,SNR_avr(1,18),1,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(8.3,SNR_avr(1,18),NaN,SNR_err(1,18),'k')
text(7.5,SNR_avr(1,18)+SNR_err(1,18)+0.5,num2str(round(SNR_avr(1,18),2)))
suptitle('ASSR : Contralateral - filtfilt, CF-1000Hz, MF-40Hz')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
%%
% cd('D:\NEC lab\Ear-Biosignal\Figure3\ASSR\ASSR_cont')
% figure(1),saveas (gcf, 'Gavr_1000_40_ver1-1.jpg')

%% SNR 1-2

spect = 35:1:45;
xscale = [35 50];
% yscale = [0 inf]; %[0 40];
yscale2 = [0 0.4];

Rspnum = [8 18 27 36 44 34 25 16 17];
Lspnum = [2 10 19 28 38 30 21 12 11];

figure(2)
for ch=1:8
    yscale = [0 ceil((pR_gavr{ch}(6,1)+pR_err{ch}(6,1)+0.05)*10^1)/10^1];
    subplot(5,9,Rspnum(ch));    
    errorbar(40,pR_gavr{ch}(6,1),NaN,pR_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pR_gavr{ch}, 'LineWidth', 2);
    xlim(xscale); ylim(yscale); grid on    
    line([40 40],[0 pR_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([45 45],[0 2],'LineWidth',0.5, 'Color', 'k')
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
    bar(8.3,SNR_avr(1,ch),1,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(8.3,SNR_avr(1,ch),NaN,SNR_err(1,ch),'k')
    text(7.5,SNR_avr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_avr(1,ch),2)))
    
    yscale = [0 ceil((pL_gavr{ch}(6,1)+pL_err{ch}(6,1)+0.05)*10^1)/10^1];
    subplot(5,9,Lspnum(ch));    
    errorbar(40,pL_gavr{ch}(6,1),NaN,pL_err{ch}(6,1),'Color',[0 0.4470 0.7410]); hold on;
    plot(f,pL_gavr{ch}, 'LineWidth', 2);
    xlim(xscale);ylim(yscale); grid on
    line([40 40],[0 pL_gavr{ch}(6,1)],'LineWidth',1, 'Color', 'r')
    line([45 45],[0 2],'LineWidth',0.5, 'Color', 'k')
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
    ax1.YAxis.TickValues = [0 yscale(1,2)/2 yscale(1,2)];
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
    bar(8.3,SNR_avr(1,ch+8),1,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(8.3,SNR_avr(1,ch+8),NaN,SNR_err(1,ch+8),'k')
    text(7.5,SNR_avr(1,ch+8)+SNR_err(1,ch+8)+0.5,num2str(round(SNR_avr(1,ch+8),2)))
end
subplot(5,9,23);
errorbar(40,pCz_gavr(6,1),NaN,pCz_err(6,1),'Color',[0 0.4470 0.7410]); hold on;
plot(f,pCz_gavr, 'LineWidth', 2);
xlim(xscale); ylim(yscale2); grid on
line([40 40],[0 pCz_gavr(6,1)],'LineWidth',1, 'Color', 'r')
line([45 45],[0 1],'LineWidth',0.5, 'Color', 'k')
title('Cz','FontWeight','bold','FontSize',12);
ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
bar(8.3,SNR_avr(1,17),1,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(8.3,SNR_avr(1,17),NaN,SNR_err(1,17),'k')
text(7.5,SNR_avr(1,17)+SNR_err(1,17)+0.5,num2str(round(SNR_avr(1,17),2)))

subplot(5,9,32);
errorbar(40,pPz_gavr(6,1),NaN,pPz_err(6,1),'Color',[0 0.4470 0.7410]); hold on;
plot(f,pPz_gavr, 'LineWidth', 2);
xlim(xscale); ylim(yscale2); grid on
line([40 40],[0 pPz_gavr(6,1)],'LineWidth',1, 'Color', 'r')
line([45 45],[0 1],'LineWidth',0.5, 'Color', 'k')
title('Pz','FontWeight','bold','FontSize',12);
ax1 = gca; ax1.XTick = [35 40 45 47.5]; ax1.XTickLabel = {'35', '40', '45', 'SNR'};
ax1.YAxis.TickValues = [0 yscale2(1,2)/2 yscale2(1,2)];
ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
ax2.XAxis.Limits = [0 10]; ax2.YAxis.Limits = [0 6]; ax2.YTick = [0 3 6]; hold on;
bar(8.3,SNR_avr(1,18),1,'FaceColor',[0.9290 0.6940 0.1250]);
errorbar(8.3,SNR_avr(1,18),NaN,SNR_err(1,18),'k')
text(7.5,SNR_avr(1,18)+SNR_err(1,18)+0.5,num2str(round(SNR_avr(1,18),2)))
suptitle('ASSR : Contralateral - filtfilt, CF-1000Hz, MF-40Hz')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

%%
cd('D:\NEC lab\Ear-Biosignal\Figure\Figure4\ASSR_Gavr')
figure(1),saveas (gcf, 'ASSR_cont_com.jpg')
figure(2),saveas (gcf, 'ASSR_cont_ind.jpg')

