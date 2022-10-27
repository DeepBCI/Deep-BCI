% segment2spectro.m
%
% calculate spectrogram
%
% author: Young-Seok Kweon
% created: 2021.11.04
%% init
clc; clear; close all;
%% mat2segment

path = '1_segment\';
save_path = '2_spectrogram\';
type = 'all\';
list = dir([path,'*.mat']);

fs = 250;

for i=1:length(list)
    load([path,list(i).name]);
    name = list(i).name(1:end-4);
    for j=1:5
        temp=AS{j};
        all=[];
        psd_mean=[];
        for ch = 1:size(temp,1)
            [s,f,t,p] = spectrogram(temp(ch,:),512,256,0.5:0.1:40,fs);
            all(:,:,ch)=p;
            fprintf('CH %d: %.2f/',ch,mean(p,'all'));
            psd_mean(ch) = mean(p,'all');
        end
        fprintf('\n');
        AS_SPEC{j}=all;
        ax = gca;
        all(:,:,psd_mean>mean(psd_mean)*2)=[];
        imagesc(t,f,mean(all,3));
        ax.CLim=[0,20];
        ax.YDir='normal';
        colorbar
        colormap(flipud(hot))
        savefig([save_path,type,'fig\',name,'_AS',num2str(j),'.fig'])
        saveas(gcf,[save_path,type,'png\',name,'_AS',num2str(j),'.png'])
    end 
    save([save_path,type,name], 'AS_SPEC','ASMT','NAME','CH','f','t','-v7.3');
    fprintf([NAME,' Done!\n']);
end