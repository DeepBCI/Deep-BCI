% PSD_1_calc.m
%
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
%% load data
fs = 250;

range = {[1 3.5],[4 7.5],[8 13.5],[14 29.5],[30 50]}; % Delta, Theta, Alpha, Beta, Gamma

removal = [9,11,21,24,26,33,35]; % non resting-state

for t=1:3
    for n=1:37
        if sum(n == removal) == 1
            continue;
        end
        
        load([path 'Winter_2023\Analysis\0_REST_no\sub',num2str(n) '_2D_' num2str(t)]);
        
        % power spectral density
        for c=1:length(CH)
            x = DATA_2D(c,:)';
            [X1, f] = periodogram(x,rectwin(size(x,1)),size(x,1), fs);
            for tr = 1:size(range,2)
                Data_PSD(c,tr) = 10*log10(bandpower(X1, f, range{tr}, 'psd'));
            end
        end
        save([path 'Winter_2023\Analysis\1_PSD_no\sub' num2str(n) '_PSD_' num2str(t)],'Data_PSD')
        fprintf(['Sub',num2str(n),'_PSD Done!\n']);        
    end
end
