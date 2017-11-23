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
cp=cvpartition(data_training, 'kFold',5);
for nu=1:5
    f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];tf_ec1=[];tf_ec2=[];tf_ec3=[];train=[];test=[];%gt=[];gtt=[];
    n_tf_ec1=[];n_tf_ec2=[];n_tf_ec3=[];n_ec1=[];n_ec2=[];n_ec3=[];ll=[];rr=[];e11=[];e22=[];e33=[];cc_rr_e1=[];
    cc_rr_e2=[];cc_rr_e3=[];cro1=[];cro2=[];cro3=[];c2ro1=[];c2ro2=[];c2ro3=[];eegtrain=[];labeltrain=[];labeltest=[];
    eegtrainn=[];e333=[];
    train = double(cp.training(nu));
    test = double(cp.test(nu));
    gt=find(train);
    gtt=find(test);
    for k=1:length(gt)
        eegtrain(:,:,k)=eeg(:,:,gt(k));
        labeltrain(k)=true_y(gt(k));
    end
    for k=1:length(gtt)
        e33(:,:,k)=eeg(:,:,gtt(k));
        labeltest(k)=true_y(gtt(k));
    end
    
    chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93];
    
    for k=1:18
        eegtrainn(k,:,:)=eegtrain(chn(k),:,:);
        e333(k,:,:)=e33(chn(k),:,:);
    end
    
    st=1;
    stt=1;
    for k=1:length(gt)
        if labeltrain(k)==1
            ll(st)=k;
            st=st+1;
        else
            rr(stt)=k;
            stt=stt+1;
        end
    end
    l=length(ll);
    r=length(rr);
    tr=length(gtt);
    free=min(l,r)-1;
    for k=1:length(ll)
        e11(:,:,k)=eegtrainn(:,:,ll(k));
    end
    for k=1:length(rr)
        e22(:,:,k)=eegtrainn(:,:,rr(k));
    end
    cq=0;
    %% STFT_region_filtering
    for si=0:20 % 0~20 angle rotation matrix
        cq=cq+1;
        ccq=0;
        for ssi=1:1
            ccq=ccq+1;
            [bbb,aaa]=butter(4,[12/50 30/50]);
            f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];
            n_ec1=[];n_ec2=[];n_ec3=[];cc_rr_e1=[];ect1=[];ect2=[];ect3=[];
            cc_rr_e2=[];cc_rr_e3=[];mask_1=[];mask_2=[];mask_3=[];result=[];
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
                    ect3(k,:)=filtfilt(bbb,aaa,e333(node,:,k));
                end
                n_ec3(:,:,node)=ect3(:,51:340);
            end
            aa=(0)*pi/180;
            ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
            aa1=(si)*pi/180;
            ro1=[cos(aa1) -sin(aa1);sin(aa1) cos(aa1)];
            aa2=(0)*pi/180;
            ro2=[cos(aa2) -sin(aa2);sin(aa2) cos(aa2)];
            for trial=1:l
                for kk=1:18
                    for k=1:18
                        cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                        cc_rc1=ro'*cc_r*ro;
                        cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                        cc_rr_e1(kk,k,trial)=cc_rr;
                        if kk==6|| k==6 % discriminated channel according to entropy
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
                        if kk==6|| k==6
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
                        if kk==6|| k==6
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
                for k=1:18
                    for kk=k+1:18
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
                for k=1:18
                    for kk=k+1:18
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
                for k=1:18
                    for kk=k+1:18
                        if teg3(k,kk)~=0
                            f3(qw)=teg3(k,kk);
                            qw=qw+1;
                        end
                    end
                end
                c_f3(trial,:)=f3;
            end
            %% compute accy
            vf=[c_f1;c_f2];
            lvf=[ones(l,1);ones(r,1)+1];
            vft=c_f3;
            options.MaxIter = 100000;
            SVMStruct = svmtrain(vf,lvf,'Options', options);
            result = svmclassify(SVMStruct,vft);
            correct=0;
            for k=1:length(result)
                if result(k)==labeltest(k)
                    correct=correct+1;
                end
            end
            accy(cq,nu,ccq)=correct/length(result)*100;
        end
    end
end
[d,dd]=max(mean(accy,2));
cv_best_rotation_matrix_angle=(dd-1)