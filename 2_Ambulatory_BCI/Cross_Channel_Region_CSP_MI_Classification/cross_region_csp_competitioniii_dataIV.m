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
chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93]; %18 channel selection
for k=1:18
    e11(k,:,:)=ee1(chn(k),:,:);
    e22(k,:,:)=ee2(chn(k),:,:);
    e33(k,:,:)=ee3(chn(k),:,:);
end
%% bandpass filtering % time segment
for cht=4:4 % # of channels in sub-region
    [bbb,aaa]=butter(4,[9/50 30/50]);
    vf=[];n_ec1=[];n_ec2=[];n_ec3=[];ect1=[];ect2=[];ect3=[];result=[];fe1=[];fe2=[];fe3=[];result1=[];stemp=[];sstemp=[];
    ssstemp=[];co1=[];co2=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for node=1:18
        for k=1:l
            ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
        end
        n_ec1(:,:,node)=ect1(:,51:300);
        for k=1:r
            ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
        end
        n_ec2(:,:,node)=ect2(:,51:300);
        for k=1:tr
            ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
        end
        n_ec3(:,:,node)=ect3(:,51:300);
    end
    si=0;
    si=si+1;
    %% compute CSP filters, m=1
    for ref=1:18 %ref=target channel
        op=0;
        for ki=1:(18-cht)
            op=op+1;
            for trial=1:l
                stemp(:,:)=n_ec1(trial,:,ref)';
                sstemp(:,:)=n_ec1(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                co1(:,:,trial,op)=ssstemp'*ssstemp/(trace(ssstemp'*ssstemp));
            end
            for trial=1:r
                stemp(:,:)=n_ec2(trial,:,ref)';
                sstemp(:,:)=n_ec2(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                co2(:,:,trial,op)=ssstemp'*ssstemp/(trace(ssstemp'*ssstemp));
            end
            for trial=1:tr
                stemp(:,:)=n_ec3(trial,:,ref)';
                sstemp(:,:)=n_ec3(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                co3(:,:,trial,op)=ssstemp'*ssstemp/(trace(ssstemp'*ssstemp));
            end
        end
        ji=-1;
        for area=1:op
            ji=ji+2;
            co11(:,:,:)=co1(:,:,:,area);
            co22(:,:,:)=co2(:,:,:,area);
            rm=mean(co11,3);
            rma=mean(co22,3);
            R=(rm+rma);
            [U,Lambda] = eig(R);
            P=sqrt(inv((Lambda)))*U';
            S{1}=P*rm*P';
            S{2}=P*rma*P';
            q=P*rm*P';
            qq=P*rma*P';
            check=(q+qq);
            [C,CC]=eig(q);
            [Ca,CCa]=eig(qq);
            nw=P'*C;
            BB=nw'*rm*nw;
            BBB=nw'*rma*nw;
            [amp1,loc1]=max(diag(BB));
            [amp2,loc2]=max(diag(BBB));
            BB(loc1,loc1)=0;
            BBB(loc2,loc2)=0;
            [amp3,loc3]=max(diag(BB));
            [amp4,loc4]=max(diag(BBB));
            wn=nw';
            hj(ji:ji+1,:)=[wn(loc1,:);wn(loc2,:)];
        end
        %% CSP feature
        opp=0;
        ji=-1;
        for ki=1:(18-cht)
            ji=ji+2;
            opp=opp+1;
            for trial=1:l
                stemp(:,:)=n_ec1(trial,:,ref)';
                sstemp(:,:)=n_ec1(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                fe1=hj(ji:ji+1,:)*ssstemp';
                max_f_l(trial,opp)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
                min_f_l(trial,opp)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
            end
        end
        opp=0;
        ji=-1;
        for ki=1:(18-cht)
            ji=ji+2;
            opp=opp+1;
            for trial=1:r
                stemp(:,:)=n_ec2(trial,:,ref)';
                sstemp(:,:)=n_ec2(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                fe2=hj(ji:ji+1,:)*ssstemp';
                max_f_r(trial,opp)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
                min_f_r(trial,opp)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
            end
        end
        opp=0;
        ji=-1;
        for ki=1:(18-cht)
            ji=ji+2;
            opp=opp+1;
            for trial=1:tr
                stemp(:,:)=n_ec3(trial,:,ref)';
                sstemp(:,:)=n_ec3(trial,:,ki:ki+cht);
                ssstemp=[stemp,sstemp];
                fe3=hj(ji:ji+1,:)*ssstemp';
                max_f_t(trial,opp)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
                min_f_t(trial,opp)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
            end
        end
        lcsp=[max_f_l,min_f_l];
        rcsp=[max_f_r,min_f_r];
        tcsp=[max_f_t,min_f_t];
        %% compute accy
        vf=[lcsp;rcsp];
        lvf=[ones(l,1);ones(r,1)+1];
        lvf=[ones(l,1);ones(r,1)+1];
        options.MaxIter = 100000;
        SVMStruct = svmtrain(vf,lvf,'Options', options);
        tvl=true_y((data_training+1):end);
        result = svmclassify(SVMStruct,tcsp);
        correct=0;
        for k=1:length(result)
            if result(k)==tvl(k)
                correct=correct+1;
            end
        end
        accy(si,ref)=correct/length(result)*100;
    end
end
[d,dd]=max(accy);
nm=chn(dd);
target_channel=nfo.clab(nm)
best_accy=d