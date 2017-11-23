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
tr=280-data_training;
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
tempp=0;
[bbb,aaa]=butter(4,[12/50 30/50]);
n_ec1=[];n_ec2=[];n_ec3=[];cc_rr_e1=[];ect1=[];ect2=[];ect3=[];
for node=1:18
    for k=1:l
        ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
    end
    n_ec1(:,:,node)=ect1(:,51:340);
    for k=1:r
        ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
    end
    n_ec2(:,:,node)=ect2(:,51:340);
    for k=1:tr
        ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
    end
    n_ec3(:,:,node)=ect3(:,51:340);
end
%% compute entropy
pch1(:,:)=log10(var(n_ec1,0,2));
pch2(:,:)=log10(var(n_ec2,0,2));
tpch=[pch1;pch2];
uch1=mean(pch1);
uch2=mean(pch2);
vch1=std(pch1,0,1);
vch2=std(pch2,0,1);
kch1=exp( (-(tpch-uch1).^2)./(2*vch1));
kch2=exp( (-(tpch-uch2).^2)./(2*vch2));
re=sum( (kch1.*log(kch1./kch2))+(kch2.*log(kch2./kch1)) );
if max((re))>tempp
    tempp=max((re));
    [d,dd]=max(re);
end
%%
max_entropy=d
nm=chn(dd);
best_discriminative_channel=nfo.clab(nm)