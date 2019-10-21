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
clear all; clc;


for i = 1:36
    str1 = 'image/Rest/Rest_S';
    str2 = int2str(i);
    str3 = '.png';
    cat1 = strcat(str1, str2);
    filename = (strcat(cat1, str3));
    %input = csvread(filename);
    
    w = imread(filename);
    w = imresize (w, [220, 250]);
    
    %figure()
    figs = imshow(w);

    for k=1:length(figs)
    
        str4 = 'resized/Rest/Rest_S';
        str5 = int2str(i);
        str6 = '.png';
        cat3 = strcat(str4, str5);
        cat4 = strcat(cat3, str6);
        saveas(figs(k), cat4)
    end

end


for i = 1:36
    str1 = 'image/Task/Task_S';
    str2 = int2str(i);
    str3 = '.png';
    cat1 = strcat(str1, str2);
    filename2 = (strcat(cat1, str3));
    %input2 = csvread(filename2);
    
    w2 = imread(filename2);
    w2 = imresize (w2, [220, 250]);
    
    figure()
    figs2 = imshow(w2);
  
    for b=1:length(figs2)
    
        str7 = 'resized/Task/Task_S';
        str8 = int2str(i);
        str9 = '.png';
        cat5 = strcat(str7, str8);
        cat6 = strcat(cat5, str9);
        saveas(figs2(b), cat6)
    end
end