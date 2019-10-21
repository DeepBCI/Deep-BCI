%% EMG: Creating an image 
% % 2019-07-18 ~  by SuJin Bak
% We aim to show pre-processing process using the EMG open data set provided by UCI.
% This is a tutorial to reduce noise for 36 people.
% In particular, it proceed segmentation about Rest vs. Task(hand gesture).
% Reference: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures

%1) Elimination of power line (60Hz) using notch filter
%2) Elimination of undesired frequency component such as DC offset, drifit:
%   1-200Hz bandpass filterfing (8th order butterworth) or 20-500Hz  -->
%   30Hz lowpass filtering
%3) To extract the envelope, abs(filted signals) and a moving average
%filter (window size: 100)--> The moving average filter is a way to smooth out changes in smoothen data to make it easier to see trends in data, eliminating unimportant noise from the data and identifying the patterns.
%4) Data are normalized per channel and down-sampled by a factor of 100 (10samples/s)

%%
%clear all; clc;


for i = 1:36
    str1 = 'Normalized_EMG/Rest/Rest_S';
    if i < 10 
        str2 = ['0',int2str(i)];
    else
        str2 = int2str(i);
    end
   
    str3 = '.csv';
    cat1 = strcat(str1, str2);
    filename = (strcat(cat1, str3));
    input = csvread(filename);
    
    Fs = 1000; %sampling frequency
    t = (0:length(input)-1)/Fs;
    
    n = normalize(t, 'range');
    n2= normalize(input, 'range');
    
    figure()
    plot(n, n2)
    
    xlim([0 1]);
    ylim([0 1]);
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


for i = 1:36
    str1 = 'Normalized_EMG/Task/Task_S';
    if i < 10 
        str2 = ['0',int2str(i)];
    else
        str2 = int2str(i);
    end
    
    str3 = '.csv';
    cat1 = strcat(str1, str2);
    filename = (strcat(cat1, str3));
    input = csvread(filename);
    
    
    Fs = 1000; %sampling frequency
    t = (0:length(input)-1)/Fs;
    
    n = normalize(t, 'range');
    n2= normalize(input, 'range');
    
    figure()
    plot(n, n2);
    xlim([0 1]);
    ylim([0 1]);
    axis square;
    
    xticks([]);
    yticks([]);
    set(gca, 'Visible', 'off');
    
    
    str4 = 'image/Task/Task_S';
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