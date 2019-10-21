%% EMG: Pre-processing
% % 2019-07-22 ~  by SuJin Bak
% We aim to show pre-processing process using the EMG open data set provided by UCI.
% This is a tutorial to reduce noise for 36 people.
% In particular, it proceed segmentation about Rest vs. Task(hand gesture).
% Reference: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures

%1) Elimination of undesired frequency component such as DC offset, drifit:0.5-1Hz bandpass filterfing (2th order butterworth)
%2) 8 channel averaging
%2) Data are normalized per channel : Root Mean Square (RMS)
% Likewise, Task state process --> Writing in Normalizaed_EMG

%%
clear all; clc;

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
    
    % Normalization: RMS
    rms_result1=ones(length(result1), 1);
    for f=1:length(result1)
        rms_result1(f) = rms(result1(f));
    end
    
    % Writing a normalized Rest data 
    str4 = 'Normalized_EMG/Rest/Rest_S';
    if i < 10 
        str5 = ['0',int2str(i)];
    else
        str5 = int2str(i);
    end

    str6 = '.csv';
    cat3 = strcat(str4, str5);
    cat4 = strcat(cat3, str6);
    csvwrite(cat4, rms_result1);
end
    

%% Reading the data (Task)

for i = 1:36
    str7 = 'EMG_data/session1/Task(1)/T';
    if i < 10 
        str8 = ['0',int2str(i)];
    else
        str8 = int2str(i);
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
    
    for u = 1:length(y2)
        sum = 0;
        for p = 1:8
            sum = sum + y2(u, p);
        end
        sum = sum/8;
        result2(u) = sum;
    end
    
    % Normalization: RMS
    rms_result2=ones(length(result2), 1);
    for f=1:length(result2)
        rms_result2(f) = rms(result2(f));
    end

   
    %Writing a normalized Task data 
    str10 = 'Normalized_EMG/Task/Task_S';
    if i < 10 
        str11 = ['0',int2str(i)];
    else
        str11 = int2str(i);
    end

    str12 = '.csv';
    cat6 = strcat(str10, str11);
    cat7 = strcat(cat6, str12);
    csvwrite(cat7, rms_result2);
end
      
  