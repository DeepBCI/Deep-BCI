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
    str1 = 'image/Rest/Rest_S';
    if i < 10 
        str2 = ['0',int2str(i)];
    else
        str2 = int2str(i);
    end

    str3 = '.png';
    cat1 = strcat(str1, str2);
    filename = (strcat(cat1, str3));

    
    w = imread(filename); % 656x875x3
    [m,n, o] = size(w);
    height = m*0.1;
    width = (height*100)/85;
    w = imresize (w, [height width]);
    
    % 모듈러 연산
    if mod((height-width),3) ~= 0 
        height = height -mod(height, 3);
        width = width -mod(width, 3);
    end
    w = imresize (w, [height width]);
    %figure()
    figs = imshow(w);

  %  for k=1:length(figs)
    
        str4 = 'resized/Rest/Rest_S';
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


for j = 1:36
    str4 = 'image/Task/Task_S';
    if j < 10 
        str5 = ['0',int2str(j)];
    else
        str5 = int2str(j);
    end

    str6 = '.png';
    cat9 = strcat(str4, str5);
    filename2 = (strcat(cat9, str6));

    
    w2 = imread(filename2);
    [mm,nn, oo] = size(w2);
    height1 = mm*0.1;
    width1 = (height1*100)/85;
    w2 = imresize (w2, [height1 width1]);
    
    % 모듈러 연산
    if mod((height1-width1),3) ~= 0 
        height1 = height1 -mod(height1, 3);
        width1 = width1 -mod(width1, 3);
    end
    w2 = imresize (w2, [height1 width1]);
    
    
    figure()
    figs2 = imshow(w2);
  
    %for b=1:length(figs2)
    
        str7 = 'resized/Task/Task_S';
        if j < 10 
            str8 = ['0',int2str(j)];
        else
            str8 = int2str(j);
        end
  
        str9 = '.png';
        cat5 = strcat(str7, str8);
        cat6 = strcat(cat5, str9);
        saveas(gcf, cat6)
    %end
end