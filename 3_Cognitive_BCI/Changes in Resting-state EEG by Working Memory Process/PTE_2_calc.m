% PTE_2_calc.m
%
% Phase transfer entropy
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
%% addpath
addpath([path,'Wake_Sleep\Lib\eeglab14_1_2b\']);
addpath([path,'Wake_Sleep\Lib\brainstorm3-master\']);
eeglab;
%% load data
fs=250;
range = {[1 3.5],[4 7.5],[8 13.5],[14 29.5],[30 50]}; % Delta, Theta, Alpha, Beta, Gamma

removal = [9,11,21,24,26,33,35]; % non resting-state

method = {'scott', 'otnes'};
% method = {'otnes'};
for t=3
    for n=31
        if sum(n == removal) == 1
            continue;
        end

        load([path 'Winter_2023\Analysis\0_REST\\sub' num2str(n) '_2D_' num2str(t)]);

        EEG_REST = pop_importdata('dataformat','matlab','nbchan',[],...
            'data',DATA_2D,'srate',fs,'pnts',0,'xmin',0);     

        for r=1:size(range,2)
           BandP=pop_eegfiltnew(EEG_REST,range{r}(1),range{r}(2)); % band-pass filter
           Data=BandP.data; 
           
           % To reduce bias, dPTE is PTE normalization
           for m=1:size(method,2)
                [dPTE(:,:,m,r) PTE(:,:,m,r)]=PhaseTE_MF_re(Data',[],method{m}); %chxchxmethodxrange
           end
        end 

        save([path 'Winter_2023\Analysis\2_PTE\sub' num2str(n) '_PTE_' num2str(t)],'dPTE','PTE');
        fprintf(['Sub' ,num2str(n),' Done!\n']);        
    end
end
