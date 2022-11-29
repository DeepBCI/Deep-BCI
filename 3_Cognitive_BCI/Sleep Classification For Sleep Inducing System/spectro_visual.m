% spectro_visual.m
%
%
%
% author: Young-Seok Kweon
% created: 2021.11.05
%% init
clc; clear; close all;
%% setting

path = '2_spectrogram\all\';
save_path = '2_spectrogram\occipital\';


%% channel location
chloc={'Oz','O1','O2','POz','PO3','PO4'};

fs = 250;
%% visualization
sleep = importdata('sleep.txt');
for i=1:size(sleep,1)
    load([path,'sub',num2str(i)]);
    name = ['sub',num2str(i)];
    
    for ch = 1:length(CH)
        ch_name=CH(ch).labels;
        for k_i=1:length(chloc)
            if strcmp(ch_name,chloc{k_i})
                chidx(k_i)=ch;
            end
        end
    end
    figure;
    cnt=0;
    for j=[3,1,4,2,5]
        cnt=cnt+1;
        temp=AS_SPEC{j};
        
        subplot(5,1,cnt);
        
        ax = gca;
        imagesc(t,f,mean(temp(:,:,chidx),3));
        temp2=mean(temp(:,:,chidx),3);
        ax.CLim=[0,20];
        ax.YDir='normal';
        colorbar
        colormap(flipud(hot))
%         if sleep(str2num(name(4:end)),j)==0
%             title('Sleep')
%         else
%             title('Wake')
%         end
        %savefig([save_path,'fig\',name,'_AS',num2str(j),'.fig'])
        %saveas(gcf,[save_path,'png\',name,'_AS',num2str(j),'.png'])
    end
    savefig([save_path,'fig\',name,'.fig'])
    saveas(gcf,[save_path,'png\',name,'.png'])
    fprintf([name,':',NAME,' Done!\n']);
end
