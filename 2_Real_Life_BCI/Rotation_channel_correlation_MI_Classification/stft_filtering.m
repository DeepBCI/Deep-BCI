clear all
load('data_set_IVa_aa');
load('true_labels_aa');
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
data_training=168;
for k=1:data_training
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end

u=1;
for k=(data_training+1):280
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    ee3(:,:,u)=temp;
    u=u+1;
    temp=0;
end
st=1;
stt=1;
for k=1:data_training
    if mrk.y(k)==1
        ll(st)=k;
        st=st+1;
    else
        rr(stt)=k;
        stt=stt+1;
    end
end
l=length(ll);
r=length(rr);
free=min(l,r)-1;
tr=112;
for k=1:l
    ee1(:,:,k)=eeg(:,:,ll(k));
end
for k=1:r
    ee2(:,:,k)=eeg(:,:,rr(k));
end
chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93];
for k=1:18
    e11(k,:,:)=ee1(chn(k),:,:);
    e22(k,:,:)=ee2(chn(k),:,:);
    e33(k,:,:)=ee3(chn(k),:,:);
end
%% stft parameter 
R=40;
window=hamming(R);
N=2^11;
L=ceil(R*0.5);
overlap=R-L;
%%
[bbb,aaa]=butter(4,[2/50 40/50]);
cc_rr_e1=[];ect1=[];ect2=[];ect3=[];cc_rr_e2=[];cc_rr_e3=[];
for node=6:6 % using best discrimination channel
    for k=1:l
        ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
    end
    for k=1:r
        ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
    end
    for k=1:tr
        ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
    end
    for k=1:l
        [S1,F1,T1,P1]=spectrogram(ect1(k,:)',window,overlap,N,100,'yaxis');
        cc_rr_e1(:,:,k)=(abs(S1));
    end
    for k=1:r
        [S2,F2,T2,P2]=spectrogram(ect2(k,:)',window,overlap,N,100,'yaxis');
        cc_rr_e2(:,:,k)=(abs(S2));
    end
end
m_ce1=sum( cc_rr_e1,3)/l;
m_ce2=sum( cc_rr_e2,3)/r;
ui=abs(m_ce1-m_ce2);
st1=std(cc_rr_e1,0,3);
st2=std(cc_rr_e2,0,3);
tss=(m_ce1-m_ce2)./sqrt( (st1.^2/l+st2.^2/r) );
for k=1:1025
    for kk=1:24
        Va(k,kk)=tcdf(abs(tss(k,kk)),free);
        V(k,kk)=Va(k,kk);
        if V(k,kk)>=0.95
            V(k,kk)=1;
        else
            V(k,kk)=0;
        end
    end
end
figure(), clf
imagesc(T1,F1,(m_ce1-m_ce2).*V);
colormap(jet)
axis xy
colorbar