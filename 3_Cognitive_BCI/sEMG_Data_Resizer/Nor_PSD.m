%% EMG: Pre-processing
% % 2019-07-22 ~  by SuJin Bak
% We aim to show pre-processing process using the EMG open data set provided by UCI.
% This is a tutorial to reduce noise for 36 people.
% In particular, it proceed segmentation about Rest vs. Task(hand gesture).
% Reference: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures

%1) Elimination of undesired frequency component such as DC offset, drifit:0.5-1Hz bandpass filterfing (2th order butterworth)
%2) 8 channel averaging
%2) Data are normalized per channel : Power Spectral Density (PSD)
% Likewise, Task state process --> Writing in Normalizaed_EMG
%%
clear all; clc; close all;

% Reading the data (Rest)


for i = 1:36
    str1 = 'EMG_data/session1/Rest(0)/R';
    
    if i < 10 
        str2 = ['0',int2str(i)];
    else
        str2 = int2str(i);
    end
    
    
    str3 = '.csv';
    cat1 = strcat(str1, str2);
    filename1 = (strcat(cat1, str3));
    input1 = csvread(filename1);
    
    % Bandpass filtering
    Fs = 1000; %sampling frequency
    x= input1;
    y = bandpass(x, [0.5 1], Fs);


    % channel averaging
    result1=ones(length(y), 1);
    
    for u = 1:length(y)
        sum = 0;
        for p = 1:8
            sum = sum + y(u, p);
        end
        sum = sum/8;
        result1(u) = sum;
    end
    
    % Making a figure for a normalized(fft) Rest data 
    figure()
    n = normalize(result1, 'range');
    h1 = hamming(length(result1));
    n2 = normalize(h1, 'range');
    periodogram(n, n2);
    xlim([0 1]);
    ylim([-250 50]);
    axis square;
    
    xticks([]);
    yticks([]);
    set(gca, 'Visible', 'off');
    
    str4 = 'image/Rest/Rest_S';
    if i < 10 
        str5 = ['0',int2str(i)];
    else
        str5 = int2str(i);
    end
    
    str6 = '.png';
    cat3 = strcat(str4, str5);
    cat4 = strcat(cat3, str6);
    saveas(gcf, cat4)
end
    

%% Reading the data (Task)

for p = 1:36
    str7 = 'EMG_data/session1/Task(1)/T';
     if p < 10 
        str8 = ['0',int2str(p)];
    else
        str8 = int2str(p);
     end
    
   
    str9 = '.csv';
    cat5 = strcat(str7, str8);
    filename2 = (strcat(cat5, str9));
    input2 = csvread(filename2);
    
    % bandpass filtering
    x2= input2;
    y2 = bandpass(x2, [0.5 1], Fs);

    
   % channel averaging
    result2=ones(length(y2), 1);
    
    for u1 = 1:length(y2)
        sum = 0;
        for p1 = 1:8
            sum = sum + y2(u1, p1);
        end
        sum = sum/8;
        result2(u1) = sum;
    end
    
    % Making a figure for a normalized(fft) Rest data 
    figure('Color',[1 1 1])
    n = normalize(result2, 'range');
    h2= hamming(length(result2));
    n2 = normalize(h2, 'range');
    periodogram(n, n2)
    
    xlim([0 1]);
    ylim([-250 50]);
    axis square;
    
    xticks([]);
    yticks([]);
    set(gca, 'Visible', 'off');
    
    str10 = 'image/Task/Task_S';
    if p < 10 
        str11 = ['0',int2str(p)];
    else
        str11 = int2str(p);
    end
    str12 = '.png';
    cat6 = strcat(str10, str11);
    cat7 = strcat(cat6, str12);
    saveas(gcf, cat7);
end
      
  