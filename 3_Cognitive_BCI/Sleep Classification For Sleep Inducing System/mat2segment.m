% mat2segment.m
%
% segment EEG by trigger
%
% author: Young-Seok Kweon
% created: 2021.11.04
%% init
clc; clear; close all;
%% mat2segment

path = '0_mat\';
list = dir([path,'*.mat']);

for i=1:length(list)
    load([path,list(i).name]);
    
    cnt=0; idx=[];
    idx_pvt=[];
    state=[];cnt1=0;
    for j=1:size(MT,1)
        if strcmp(MT{j,2},'S 11')
            cnt=cnt+1;
            idx_pvt(cnt,1)=j;
        elseif strcmp(MT{j,2},'S 14')
            idx_pvt(cnt,2)=j;
        elseif strcmp(MT{j,2},'S 21')
            cnt1=cnt1+1;
            state(cnt1)=2;
            idx(2,1)=j;
        elseif strcmp(MT{j,2},'S 22')
            idx(2,2)=j;
        elseif strcmp(MT{j,2},'S 31')
            cnt1=cnt1+1;
            state(cnt1)=3;
            idx(3,1)=j;
        elseif strcmp(MT{j,2},'S 32')    
             idx(3,2)=j;
        elseif strcmp(MT{j,2},'S 41')
            cnt1=cnt1+1;
            state(cnt1)=4;
            idx(4,1)=j;
        elseif strcmp(MT{j,2},'S 42')
             idx(4,2)=j;
        elseif strcmp(MT{j,2},'S 51')
            cnt1=cnt1+1;
            state(cnt1)=5;
            idx(5,1)=j;
        elseif strcmp(MT{j,2},'S 52')
             idx(5,2)=j;
        elseif strcmp(MT{j,2},'S 61')
            cnt1=cnt1+1;
            state(cnt1)=6;
            idx(6,1)=j;
        elseif strcmp(MT{j,2},'S 62')
             idx(6,2)=j;
        end
    end
    
    cnt=0;
    temp_idx=[];
    for j=1:length(state)-1
        if state(j)==state(j+1)
            cnt=cnt+1;
            temp_idx(cnt)=j+1;
        end
    end
    state(temp_idx)=[];
    
    AS=[];
    ASMT=[];
    for j=1:5
        AS{j} = DATA(:,MT{int64(idx(j+1,1)),1}:MT{int64(idx(j+1,2)),1});
        ASMT{j} = MT(int64(idx(j+1,1)):int64(idx(j+1,2)),:);
    end
    
    PVT=[];
    PVTMT=[];
    for j=1:5
        PVT{state(j)-1} = DATA(:,MT{int64(idx_pvt(j,1)),1}:MT{int64(idx_pvt(j,2)),1});
        PVTMT{state(j)-1} = MT(int64(idx_pvt(j,1)):int64(idx_pvt(j,2)),:);
    end
    save(['1_segment\',list(i).name], 'AS','ASMT','PVT','PVTMT','NAME','CH');
    fprintf([list(i).name,'\n']);
end