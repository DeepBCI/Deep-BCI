clc; clear all; close all;

folder = {'191129_lys','191129_pjy','191202_ljy','191202_ssh','191203_cjh','191206_pjh','191209_chr','191209_ykh','191210_JJH','191211_KGB','191213_MIG','191218_LCB','191219_lht','191219_yyd','191220_ydj'};
for sub=1:15
    cd(['D:\NEC lab\Ear-Biosignal\Data_biosignal\' folder{sub}]')
    
    rawdata = importdata([folder{sub} '_ECEO.txt'], '\t', 2);
    
    % start tick
    fid = fopen([folder{sub} '_ECEO.txt']);
    start_time = textscan(fid, '%s');
    start = cell2mat(start_time{1,1}(end-12,1));
    
    n = 3; Fs = 250; Fn = Fs/2;
    
    [b,a]=butter(n, [59 61]/Fn, 'stop');
    stop_data = filtfilt(b,a,rawdata.data);
    [b,a]=butter(n, [0.5 100]/Fn, 'bandpass');
    bp = filtfilt(b,a,stop_data);
    
    ECEO(:,:,1) = bp(start+1000:start+16049,:);
    ECEO(:,:,2) = bp(start+16100:start+31149,:);
    ECEO(:,:,3) = bp(start+31200:start+46249,:);
    
    for ch=1:8
        Rear_ch{ch} = ECEO(1:15050,10+ch,:)-ECEO(1:15050,19,:);
        Lear_ch{ch} = ECEO(1:15050,ch,:)-ECEO(1:15050,9,:);
    end
    for ch=1:3
        Scalp_ch{ch} = ECEO(1:15050,ch+29,:);
    end
    
    EC_s1 = 1; EC_s2 = 7500;
    EO_s1 = 7551; EO_s2 = 15050;
    spect = 9:1:11;
    win = 250;
    overlap = 125;
    
    % SNR
    for j = 1:3
        for ch=1:8
            [~, ~, ~, ECpR_ch{sub,ch}(:,:,j)] = spectrogram(Rear_ch{ch}(EC_s1:EC_s2,:,j), win, overlap, spect, Fs);
            [~, ~, ~, ECpL_ch{sub,ch}(:,:,j)] = spectrogram(Lear_ch{ch}(EC_s1:EC_s2,:,j), win, overlap, spect, Fs);
            [~, ~, ~, EOpR_ch{sub,ch}(:,:,j)] = spectrogram(Rear_ch{ch}(EO_s1:EO_s2,:,j), win, overlap, spect, Fs);
            [~, ~, ~, EOpL_ch{sub,ch}(:,:,j)] = spectrogram(Lear_ch{ch}(EO_s1:EO_s2,:,j), win, overlap, spect, Fs);
        end
        for ch=1:3
            [~, ~, ~, ECpS_ch{sub,ch}(:,:,j)] = spectrogram(Scalp_ch{ch}(EC_s1:EC_s2,:,j), win, overlap, spect, Fs);
            [s, f, t, EOpS_ch{sub,ch}(:,:,j)] = spectrogram(Scalp_ch{ch}(EO_s1:EO_s2,:,j), win, overlap, spect, Fs);
        end
    end
    
    for ch=1:8
        SNR(sub,ch) = mean(ECpR_ch{sub,ch},'all')/mean(EOpR_ch{sub,ch},'all');
        SNR(sub,8+ch) = mean(ECpL_ch{sub,ch},'all')/mean(EOpL_ch{sub,ch},'all');
    end
    for ch=1:3
        SNR(sub,ch+16) = mean(ECpS_ch{sub,ch},'all')/mean(EOpS_ch{sub,ch},'all');
    end
    
    % figure
    spect2 = 1:1:30;    
    
    for j = 1:3
        for ch=1:8
            [~, ~, ~, pR_ch{sub,ch}(:,:,j)] = spectrogram(Rear_ch{ch}(:,:,j), win, overlap, spect2, Fs);
            [~, ~, ~, pL_ch{sub,ch}(:,:,j)] = spectrogram(Lear_ch{ch}(:,:,j), win, overlap, spect2, Fs);
        end
        for ch=1:3
            [s, f, t, pS_ch{sub,ch}(:,:,j)] = spectrogram(Scalp_ch{ch}(:,:,j), win, overlap, spect2, Fs);
        end
    end
    
    for ch=1:8
        pR_avr{ch}(:,:,sub) = mean(pR_ch{sub,ch},3);
        pL_avr{ch}(:,:,sub) = mean(pL_ch{sub,ch},3);
    end
    for ch=1:3
        pS_avr{ch}(:,:,sub) = mean(pS_ch{sub,ch},3);
    end    
end

for ch=1:8
    pR_gavr{ch} = mean(pR_avr{ch},3);    pL_gavr{ch} = mean(pL_avr{ch},3);
    pR_err{ch} = std(pR_avr{ch},0,3)/sqrt(15); pL_err{ch} = std(pL_avr{ch},0,3)/sqrt(15);    
end
for ch=1:3
    pS_gavr{ch} = mean(pS_avr{ch},3); pS_err{ch} = std(pS_avr{ch},0,3)/sqrt(15);    
end

SNR_avr = mean(SNR,1);
SNR_err = std(SNR)/sqrt(15);

%% SNR 1-1
close all;

% yscale = [0 0.5];
% yscale2 = [0 3];
xscale = [0 80];
scale = [0 0.5];
scale2 = [0 10];

Rspnum = [8 18 27 36 44 34 25 16 17];
Lspnum = [2 10 19 28 38 30 21 12 11];
Sspnum = [40 41 42];
Sname = {'O1','Oz','O2'};

figure(1)
for ch=1:8
    subplot(5,9,Rspnum(ch));
    imagesc(t, f, pR_gavr{ch});
    set(gca,'yDir','normal')
    caxis(scale); xlim(xscale);
    line([60 60],[0 30],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [0 30 60 73]; ax1.XTickLabel = {'0', '30', '60', 'SNR'};    
    ax1.FontSize = 8; ax1.FontWeight = 'bold'; ax1.XColor = 'k'; ax1.YColor = 'k'; 
    title(['R' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 80]; ax2.YAxis.Limits = [0 7]; ax2.YTick = [0 3.5 7]; hold on;
    bar(70,SNR_avr(1,ch),8,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(70,SNR_avr(1,ch),SNR_err(1,ch),SNR_err(1,ch),'k')
    text(62.5,SNR_avr(1,ch)+SNR_err(1,ch)+0.5,num2str(round(SNR_avr(1,ch),2)),'FontWeight','bold','FontSize',8)
    ax2.FontSize = 8; ax2.FontWeight = 'bold'; ax2.YColor = 'k';
end
caxis(scale); 
hb = colorbar('location','eastoutside','Ticks',[0 scale(1,2)/2 scale(1,2)],'FontSize',9,'FontWeight','bold','Color','k');
set(hb,'Units','normalized', 'position', [0.93 0.27 0.02 0.425]);
% set(hb,'Units','normalized', 'position', [0.78 0.268 0.018 0.265]);
% set(hb,'Units','normalized', 'position', [0.62 0.27 0.02 0.425]);

% hb = colorbar('location','southoutside','Ticks',[0 0.75 1.5],'FontSize',9,'FontWeight','bold','Color','k');
% set(hb,'Units','normalized', 'position', [0.751 0.055 0.065 0.025]); % colorbar bottom

for ch=1:8
    subplot(5,9,Lspnum(ch));
    imagesc(t, f, pL_gavr{ch});
    set(gca,'yDir','normal')
    caxis(scale); xlim(xscale);
    line([60 60],[0 30],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [0 30 60 73]; ax1.XTickLabel = {'0', '30', '60', 'SNR'};    
    ax1.FontSize = 8; ax1.FontWeight = 'bold'; ax1.XColor = 'k'; ax1.YColor = 'k'; 
    title(['L' num2str(ch)],'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 80]; ax2.YAxis.Limits = [0 7]; ax2.YTick = [0 3.5 7]; hold on;
    bar(70,SNR_avr(1,ch+8),8,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(70,SNR_avr(1,ch+8),SNR_err(1,ch+8),SNR_err(1,ch+8),'k')
    text(62.5,SNR_avr(1,ch+8)+SNR_err(1,ch+8)+0.5,num2str(round(SNR_avr(1,ch+8),2)),'FontWeight','bold','FontSize',8)
    ax2.FontSize = 8; ax2.FontWeight = 'bold'; ax2.YColor = 'k';
end
caxis(scale); 
hb2 = colorbar('location','westoutside','Ticks',[0 scale(1,2)/2 scale(1,2)],'FontSize',9,'FontWeight','bold','Color','k');
set(hb2,'Units','normalized', 'position', [0.088 0.27 0.02 0.425]);
% set(hb2,'Units','normalized', 'position', [0.24 0.268 0.018 0.265]);
% set(hb2,'Units','normalized', 'position', [0.4 0.27 0.02 0.425]);
% hb2 = colorbar('location','southoutside','Ticks',[0 0.35 0.7],'FontSize',9,'FontWeight','bold','Color','k');
% set(hb2,'Units','normalized', 'position', [0.219 0.055 0.065 0.025]); % colorbar bottom

for ch=1:3
    subplot(5,9,Sspnum(ch));
    imagesc(t, f, pS_gavr{ch});
    set(gca,'yDir','normal')
    caxis(scale2); xlim(xscale);
    line([60 60],[0 30],'LineWidth',0.5, 'Color', 'k')
    ax1 = gca; ax1.XTick = [0 30 60 73]; ax1.XTickLabel = {'0', '30', '60', 'SNR'};    
    ax1.FontSize = 8; ax1.FontWeight = 'bold'; ax1.XColor = 'k'; ax1.YColor = 'k'; 
    title(Sname{ch},'FontWeight','bold','FontSize',12);
    ax2 = axes('Position',ax1.Position, 'XAxisLocation','top', 'YAxisLocation','right','Color','none');
    ax2.YAxis.Visible = 'on'; ax2.XAxis.Visible = 'off';
    ax2.XAxis.Limits = [0 80]; ax2.YAxis.Limits = [0 7]; ax2.YTick = [0 3.5 7]; hold on;
    bar(70,SNR_avr(1,ch+16),8,'FaceColor',[0.9290 0.6940 0.1250]);
    errorbar(70,SNR_avr(1,ch+16),SNR_err(1,ch+16),SNR_err(1,ch+16),'k')
    text(62.5,SNR_avr(1,ch+16)+SNR_err(1,ch+16)+0.5,num2str(round(SNR_avr(1,ch+16),2)),'FontWeight','bold','FontSize',8)
    ax2.FontSize = 8; ax2.FontWeight = 'bold'; ax2.YColor = 'k';
end
caxis(scale2); hb = colorbar('location','southoutside','FontSize',9,'FontWeight','bold','Color','k');
set(hb,'Units','normalized', 'position', [0.485 0.055 0.065 0.025]); % colorbar bottom
% set(hb,'Units','normalized', 'position', [0.485 0.25 0.065 0.025]); % colorbar top
supt = suptitle('ECEO : Ipsilateral - filtfilt');
set(supt,'FontSize',18,'FontWeight','bold','Color','k')
set(gcf,'units','normalized','outerposition',[0 0 1 1]);

%%
cd('D:\NEC lab\Ear-Biosignal\Figure\Figure4\ECEO_Gavr')
figure(1),saveas (gcf, 'ECEO_ipsil.jpg')