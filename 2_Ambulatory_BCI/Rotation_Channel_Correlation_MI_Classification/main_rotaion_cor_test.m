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
cq=0;
%% STFT_region_filtering
ssi=6; % discriminated channel according to entropy
[bbb,aaa]=butter(4,[12/50 30/50]); %STFT 분석으로 얻어진 주파수-시간 대역 사용
f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];
n_ec1=[];n_ec2=[];n_ec3=[];
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
aa=(0)*pi/180;
ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
aa1=(11)*pi/180; % anlge from CV
ro1=[cos(aa1) -sin(aa1);sin(aa1) cos(aa1)];
%% compute channel correlation
for trial=1:l
    for kk=1:18
        for k=1:18
            cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
            cc_rc1=ro'*cc_r*ro;
            cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
            cc_rr_e1(kk,k,trial)=cc_rr;
            if kk==ssi|| k==ssi
                cc_rc1=ro1'*cc_r*ro1;
                cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                cc_rr_e1(kk,k,trial)=cc_rr;
            end
        end
    end
end
for trial=1:r
    for kk=1:18
        for k=1:18
            cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
            cc_rc2=ro'*cc_r2*ro;
            cc_rr2=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
            cc_rr_e2(kk,k,trial)=cc_rr2;
            if kk==ssi|| k==ssi
                cc_rc2=ro1'*cc_r2*ro1;
                cc_rr=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
                cc_rr_e2(kk,k,trial)=cc_rr;
            end
        end
    end
end
for trial=1:tr
    for kk=1:18
        for k=1:18
            cc_r3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
            cc_rc3=ro'*cc_r3*ro;
            cc_rr3=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
            cc_rr_e3(kk,k,trial)=cc_rr3;
            if kk==ssi|| k==ssi
                cc_rc3=ro1'*cc_r3*ro1;
                cc_rr=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
                cc_rr_e3(kk,k,trial)=cc_rr;
            end
        end
    end
end
%% discriminated channel correlation based on T-statistic
m_ce1=sum( cc_rr_e1,3)/l;
m_ce2=sum( cc_rr_e2,3)/r;
ui=abs(m_ce1-m_ce2);
st1=std(cc_rr_e1,0,3);
st2=std(cc_rr_e2,0,3);
tss=(m_ce1-m_ce2)./sqrt( (st1.^2/l+st2.^2/r) );
for k=1:18
    for kk=1:18
        Va(k,kk)=tcdf(abs(tss(k,kk)),free);
        V(k,kk)=Va(k,kk);
        if V(k,kk)>=0.97
            V(k,kk)=1;
        else
            V(k,kk)=0;
        end
    end
end
%% feature selection
for trial=1:l
    mask_1(:,:,trial)=cc_rr_e1(:,:,trial).*V;
    teg1=mask_1(:,:,trial);
    qw=1;
    qww=1;
    for k=1:18
        for kk=k+1:18
            tot1(1,qww)=cc_rr_e1(k,kk,trial);
            qww=qww+1;
            if teg1(k,kk)~=0
                f1(qw)=teg1(k,kk);
                qw=qw+1;
            end
        end
    end
    c_f1(trial,:)=f1;
end
for trial=1:r
    mask_2(:,:,trial)=cc_rr_e2(:,:,trial).*V;
    teg2=mask_2(:,:,trial);
    qw=1;
    qww=1;
    for k=1:18
        for kk=k+1:18
            tot2(1,qww)=cc_rr_e2(k,kk,trial);
            qww=qww+1;
            if teg1(k,kk)~=0
                f2(qw)=teg2(k,kk);
                qw=qw+1;
            end
        end
    end
    c_f2(trial,:)=f2;
end
for trial=1:tr
    mask_3(:,:,trial)=cc_rr_e3(:,:,trial).*V;
    teg3=mask_3(:,:,trial);
    qw=1;
    qww=1;
    for k=1:18
        for kk=k+1:18
            tot3(1,qww)=cc_rr_e3(k,kk,trial);
            qww=qww+1;
            if teg3(k,kk)~=0
                f3(qw)=teg3(k,kk);
                qw=qw+1;
            end
        end
    end
    c_f3(trial,:)=f3;
end
%% compute accy
vf=( [c_f1;c_f2] );
vft=( c_f3);
options.MaxIter = 100000;
lvf=[ones(l,1);ones(r,1)+1];

SVMStruct = svmtrain(vf,lvf,'Options', options);
tvl=true_y(169:end);
result = svmclassify(SVMStruct,vft);
correct=0;
for k=1:length(result)
    if result(k)==tvl(k)
        correct=correct+1;
    end
end
accy=correct/length(result)*100;
