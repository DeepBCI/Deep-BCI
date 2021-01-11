clear all
load('data_set_IVa_ay');
load('true_labels_ay');
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
train=28;
for k=1:train
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end
u=1;
for k=(train+1):280
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    ee3(:,:,u)=temp;
    u=u+1;
    temp=0;
end
st=1;
stt=1;
for k=1:train
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
tr=280-train;
for k=1:l
    ee1(:,:,k)=eeg(:,:,ll(k));
end
for k=1:r
    ee2(:,:,k)=eeg(:,:,rr(k));
end
chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93];%used channel 118channel -> 18channel
cl=18;
for k=1:cl
    e11(k,:,:)=ee1(chn(k),:,:);
    e22(k,:,:)=ee2(chn(k),:,:);
    e33(k,:,:)=ee3(chn(k),:,:);
end
[bbb,aaa]=butter(4,[7/50 30/50]);
%% FFT
for node=1:cl
    
    for k=1:l
        ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
    end
    n_ec1(:,:,node)=ect1(:,51:300);
    for trial=1:l
        ch1(trial,:,node)=(abs( (fft(n_ec1(trial,:,node)))/sqrt(250))); % FFT of EEG signal
    end
    for k=1:r
        ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
    end
    n_ec2(:,:,node)=ect2(:,51:300);
    for trial=1:r
        ch2(trial,:,node)=(abs( (fft(n_ec2(trial,:,node)) )/sqrt(250)));
    end
    for k=1:tr
        ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
    end
    n_ec3(:,:,node)=ect3(:,51:300);
    for trial=1:tr
        ch3(trial,:,node)=(abs( (fft(n_ec3(trial,:,node)) )/sqrt(250)));
    end
end
%%
for trial=1:l
    leg1(:,:)=n_ec1(trial,:,:);
    o1(:,:,trial)=leg1'*leg1/trace(leg1'*leg1);
end
for trial=1:r
    leg2(:,:)=n_ec2(trial,:,:);
    o2(:,:,trial)=leg2'*leg2/trace(leg2'*leg2);
    
end
for trial=1:tr
    leg3(:,:)=n_ec3(trial,:,:);
    o3(:,:,trial)=leg3'*leg3/trace(leg3'*leg3);
    
end
[rt_1,diff1_1]=ykcsp(o1,o2);
rt_1(1,1:18)=1;
%% Frequency weight
cl=1;
fl=0;
for rangee=60:60
    fl=0;
    leg1_ch_1=[];leg1_ch_2=[];leg1_ch_3=[];co1=[];co2=[];co3=[];cov_f_l=[];cov_f_r=[];hj=[];hj2=[];nop=[];nopp=[];hjj=[];hjj2=[];n1_ch1=[];n1_ch2=[];n1_ch3=[];no11=[];no22=[];no33=[];
    nno11=[];nno22=[];nno33=[];nch1=[];nch2=[];nch3=[];nnch1=[];nnch2=[];nnch3=[];
    range=71-rangee;
    for fle=25:(range+1):100 % frequency range for optimal weighting
        fl=fl+1;
        for trial=1:l
            leg1_ch_1(:,:)=ch1(trial,fle:fle+range,:);
            co1(:,:,trial)=leg1_ch_1'*leg1_ch_1/trace(leg1_ch_1'*leg1_ch_1);
        end
        cov_f_l(:,:,fl)=mean(co1,3);
    end
    fl=0;
    for fle=25:(range+1):100
        fl=fl+1;
        for trial=1:r
            leg1_ch_2(:,:)=ch2(trial,fle:fle+range,:);
            co2(:,:,trial)=leg1_ch_2'*leg1_ch_2/trace(leg1_ch_2'*leg1_ch_2);
        end
        cov_f_r(:,:,fl)=mean(co2,3);
    end
    %% Optimal frequency weight filter
    for k=1:fl
        ap1(:,:)=(cov_f_l(:,:,k)*rt_1(1,:)'*rt_1(1,:)*cov_f_l(:,:,k));
        ap2(:,:)=(cov_f_r(:,:,k)*rt_1(1,:)'*rt_1(1,:)*cov_f_r(:,:,k));
        [e1,e2]=eig(pinv(ap2)*ap1);
        [e3,e4]=eig(pinv(ap1)*ap2);
        R=(ap1+ap2);
        [U,Lambda] = eig(R);
        P=sqrt(pinv((Lambda)))*(U).';
        q=P*ap1*P.';
        qq=P*ap2*P.';
        check=(q+qq);
        [C,CC]=eig(q);
        mk=max(diag(CC));
        mmk=min(diag(CC));
        
        [Ca,CCa]=eig(qq);
        nw=P.'*C;
        BB=nw.'*ap1*nw;
        BBB=nw.'*ap2*nw;
        [amp1,loc1]=max(diag(BB));
        [amp2,loc2]=max(diag(BBB));
        BB(loc1,loc1)=0;
        BBB(loc2,loc2)=0;
        [amp3,loc3]=max(diag(BB));
        [amp4,loc4]=max(diag(BBB));
        wn=nw.';
        hj(k,:)=e1(:,1);
        hj2(k,:)=e3(:,1);
    end
    %% filtered EEG signal
    trm=(range+1);
    for k=1:fl
        hjj(trm*(k-1)+1:trm*(k-1)+trm,:)=ones(trm,18).*hj(k,:);
        hjj2(trm*(k-1)+1:trm*(k-1)+trm,:)=ones(trm,18).*hj2(k,:);
    end
    for  kk=1:l
        n1_ch1(:,:)=ch1(kk,25:fle+range,:);
        no11(kk,:,:)=n1_ch1.*hjj;
    end
    for  kk=1:r
        n1_ch2(:,:)=ch2(kk,25:fle+range,:);
        no22(kk,:,:)=n1_ch2.*hjj;
    end
    for  kk=1:tr
        n1_ch3(:,:)=ch3(kk,25:fle+range,:);
        no33(kk,:,:)=n1_ch3.*hjj;
    end
    for  kk=1:l
        n1_ch1(:,:)=ch1(kk,25:fle+range,:);
        nno11(kk,:,:)=n1_ch1.*hjj2;
    end
    for  kk=1:r
        n1_ch2(:,:)=ch2(kk,25:fle+range,:);
        nno22(kk,:,:)=n1_ch2.*hjj2;
    end
    for  kk=1:tr
        n1_ch3(:,:)=ch3(kk,25:fle+range,:);
        nno33(kk,:,:)=n1_ch3.*hjj2;
    end
%%
    for node=1:18
        for trial=1:l
            nch1(trial,:,node)=(abs( ifft(no11(trial,:,node))));
        end
        for trial=1:r
            nch2(trial,:,node)=(abs( ifft(no22(trial,:,node))));
        end
        for trial=1:tr
            nch3(trial,:,node)=(abs( ifft(no33(trial,:,node))));
        end
    end
    for node=1:18
        for trial=1:l
            nnch1(trial,:,node)=(abs( ifft(nno11(trial,:,node))));
        end
        for trial=1:r
            nnch2(trial,:,node)=(abs( ifft(nno22(trial,:,node))));
        end
        for trial=1:tr
            nnch3(trial,:,node)=(abs( ifft(nno33(trial,:,node))));
        end
    end
    [re1_l,re1_r,re1_t,re1_d]=yk4one(l,r,tr,nch1,nch2,nch3);
    [re2_l,re2_r,re2_t,re2_d]=yk4two(l,r,tr,nnch1,nnch2,nnch3);
    clcsp=[re1_l,re2_l];
    crcsp=[re1_r,re2_r];
    ctcsp=[re1_t,re2_t];
    tvl=true_y((train+1):end);
    k1=0;
    k2=0;
    for k=1:tr
        if tvl(k)==1
            k1=k1+1;
            c1ctcsp(k1,:)=ctcsp(k,:);
        else
            k2=k2+1;
            c2ctcsp(k2,:)=ctcsp(k,:);
            
        end
    end
    
    vf=[clcsp;crcsp];
    lvf=[ones(l,1);ones(r,1)+1];
    lvf=[ones(l,1);ones(r,1)+1];
    options.MaxIter = 100000;
    SVMStruct = svmtrain(vf,lvf,'Options', options);
    mdl=fitcsvm(vf,lvf);
    
    [result1,sco]= predict(mdl,ctcsp);
    result = svmclassify(SVMStruct,ctcsp);
    correct1=0;
    for k=1:length(result1)
        if result1(k)==tvl(k)
            correct1=correct1+1;
        end
    end
    correct=0;
    for k=1:length(result)
        if result(k)==tvl(k)
            correct=correct+1;
        end
    end
    
    accy(rangee)=correct/length(result)*100;
end