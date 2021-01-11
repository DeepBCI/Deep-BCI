clear all
clear all
load('sp1s_aa_1000Hz');
cue=1;
cuee=1;
tvl=[1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,0];
tvl=tvl+1;
for k=1:316
    if y_train(k)==0;
        temp1=[];
        temp1(:,:)=x_train(146:500,:,k)';
        for kk=1:28
            x1=temp1(kk,:);
            z1 = @(x1) (x1 - mean(x1))./std(x1);
            x11(kk,:) = z1(x1) ;
        end
        eeg_1(:,:,cue)=x_train(146:500,:,k)';
        cue=cue+1;
    else
        temp1=[];
        temp1(:,:)=x_train(:,:,k)';
        for kk=1:28
            x1=temp1(kk,:);
            z1 = @(x1) (x1 - mean(x1))./std(x1);
            x22(kk,:) = z1(x1) ;
        end
        eeg_2(:,:,cuee)=x_train(146:500,:,k)';
        cuee=cuee+1;
    end
end
for k=1:100
    temp1=[];
    temp1(:,:)=x_test(:,:,k)';
    for kk=1:28
        x1=temp1(kk,:);
        z1 = @(x1) (x1 - mean(x1))./std(x1);
        x33(kk,:) = z1(x1) ;
    end
    test_d(:,:,k)=x_test(146:500,:,k)';
end
for k=1:100
    for kk=1:28
        test_dd(kk,:,k)=test_d(kk,:,k)-sum(test_d(:,:,k),1)/28;
    end
end
for k=1:159
    for kk=1:28
        eg_1(kk,:,k)=eeg_1(kk,:,k)-sum(eeg_1(:,:,k),1)/28;
    end
end
for k=1:157
    for kk=1:28
        eg_2(kk,:,k)=eeg_2(kk,:,k)-sum(eeg_2(:,:,k),1)/28;
    end
end
R=40;
window=hamming(R);
N=2^9;
L=ceil(R*0.5);
overlap=R-L;
e11=eg_1;
e22=eg_2;
e33=test_dd;
l=159;
r=157;
tr=100;
tsum1=[];
tsum2=[];
ccq=0;
for bz=5:5
    for fqq=6:1:6
        for fq=0.65:0.05:0.65
            for fqqq=0:0
                for qr=3:3
                    ccq=ccq+1;
                    [bbb,aaa]=butter(bz,[8/500 (30+qr)/500]);
                    tend=500;
                    f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];tf_ec1=[];tf_ec2=[];tf_ec3=[];
                    n_tf_ec1=[];n_tf_ec2=[];n_tf_ec3=[];n_ec1=[];n_ec2=[];n_ec3=[];cc_rr_e1=[];ect1=[];ect2=[];ect3=[];
                    cc_rr_e2=[];cc_rr_e3=[];cro1=[];cro2=[];cro3=[];c2ro1=[];c2ro2=[];c2ro3=[];mask_1=[];mask_2=[];mask_3=[];result=[];stemp=[];fe1=[];fe2=[];fe3=[];result1=[];stemp=[];sstemp=[];
                    ssstemp=[];co1=[];co2=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];tlcsp=[];trcsp=[];ttcsp=[];V=[];Va=[];VV=[];
                    for node=1:28
                        for k=1:l
                            ect1(k,:)=filtfilt(bbb,aaa,e11(node,fqq:355-fqqq,k));
                        end
                        n_ec1(:,:,node)=ect1(:,:);
                        for k=1:r
                            ect2(k,:)=filtfilt(bbb,aaa,e22(node,fqq:355-fqqq,k));
                        end
                        n_ec2(:,:,node)=ect2(:,:);
                        for k=1:tr
                            ect3(k,:)=filtfilt(bbb,aaa,e33(node,fqq:355-fqqq,k));
                        end
                        n_ec3(:,:,node)=ect3(:,:);
                    end
                    %%
                    for trial=1:l
                        for kk=1:28
                            for k=1:28
                                g1=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                                gg_rr=g1(2)/sqrt(g1(1,1)*g1(2,2));
                                gg_rr_e1(kk,k,trial)=gg_rr;
                            end
                        end
                    end
                    for trial=1:r
                        for kk=1:28
                            for k=1:28
                                g2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                                gg_rr2=g2(2)/sqrt(g2(1,1)*g2(2,2));
                                gg_rr_e2(kk,k,trial)=gg_rr2;
                            end
                        end
                    end
                    for trial=1:tr
                        for kk=1:28
                            for k=1:28
                                g3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
                                gg_rr3=g3(2)/sqrt(g3(1,1)*g3(2,2));
                                gg_rr_e3(kk,k,trial)=gg_rr3;
                            end
                        end
                    end
                    cor_check1=abs(mean(gg_rr_e1,3));
                    cor_check2=abs(mean(gg_rr_e2,3));
                    jk=0;
                    for yo=1:28
                        for yyo=yo+1:28
                            jk=jk+1;
                            coq1(1,jk)=cor_check1(yo,yyo);
                            coq2(1,jk)=cor_check2(yo,yyo);
                        end
                    end
                    coq11=sort(coq1);
                    coq22=sort(coq2);
                    hq1=coq11(1,100);
                    hq2=coq22(1,100);
                    cor_check1(cor_check1>fq)=0;
                    cor_check2(cor_check2>fq)=0;
                    VV=cor_check1.*cor_check2;
                    VV(VV>0)=1;
                    n_ec11=[];
                    n_ec22=[];
                    n_ec33=[];
                    for ch=1:28
                        [new_cl,new_cr,new_ct] =anc_ch5(l,r,tr,n_ec1,n_ec2,n_ec3,ch,VV,(355-fqqq-fqq+1));
                        n_ec11(:,:,ch)=new_cl;
                        n_ec22(:,:,ch)=new_cr;
                        n_ec33(:,:,ch)=new_ct;
                    end
                    n_ec1=[];
                    n_ec2=[];
                    n_ec3=[];
                    n_ec1=n_ec11;
                    n_ec2=n_ec22;
                    n_ec3=n_ec33;                   
%%
                    for trial=1:l
                        for kk=1:28
                            for k=1:28
                                g1=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                                gg_rr=g1(2)/sqrt(g1(1,1)*g1(2,2));
                                g1_rr_e1(kk,k,trial)=gg_rr;
                            end
                        end
                    end
                    for trial=1:r
                        for kk=1:28
                            for k=1:28
                                g2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                                gg_rr2=g2(2)/sqrt(g2(1,1)*g2(2,2));
                                g1_rr_e2(kk,k,trial)=gg_rr2;
                            end
                        end
                    end
                    for trial=1:tr
                        for kk=1:28
                            for k=1:28
                                g3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
                                gg_rr3=g3(2)/sqrt(g3(1,1)*g3(2,2));
                                g1_rr_e3(kk,k,trial)=gg_rr3;
                            end
                        end
                    end
%%
                    ch=28;
                    ssi=0;
                    si=0;
                    aa=(si)*pi/180;
                    siz=28;
                    magle=pi/4;
                    aa=zeros(siz-1,siz);
                    for si=0:1:0
                        ssi=ssi+1;
                        aa=(si)*pi/180;
                        a1a=[0:1:90]*pi/180;
                        for trial=1:l
                            for kk=1:siz
                                for k=kk+1:siz
                                    cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                                    A=(cc_r(2,2)-cc_r(1,1))/2;
                                    B=cc_r(2);
                                    magle=(pi/8)-(acos( cc_r(2)/ sqrt( cc_r(2)*cc_r(2)+ ( (cc_r(1,1)-cc_r(2,2))/2 )^2) ))/2;
                                    C=sqrt(  ( cc_r(1,1)+cc_r(2,2) )^2 - cc_r(2)*cc_r(2)  -  0.5*( (cc_r(1,1)-cc_r(2,2) )^2));
                                    X1(kk,k,trial)=real(A/C);
                                    X2(kk,k,trial)=real(B/C);
                                end
                            end
                        end
                        C1=mean(X1,3);
                        D1=mean(X2,3);
                        for trial=1:r
                            for kk=1:siz
                                for k=kk+1:siz
                                    cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                                    A2=(cc_r2(2,2)-cc_r2(1,1))/2;
                                    B2=cc_r2(2);
                                    magle=(pi/8)-(acos( cc_r2(2)/ sqrt( cc_r2(2)*cc_r2(2)+ ( (cc_r2(1,1)-cc_r2(2,2))/2 )^2) ))/2;
                                    C22=sqrt(  ( cc_r2(1,1)+cc_r2(2,2) )^2 - cc_r2(2)*cc_r2(2)  -  0.5*( ( cc_r2(2,2)-cc_r2(1,1) )^2 ));
                                    X3(kk,k,trial)=real(A2/C22);
                                    X4(kk,k,trial)=real(B2/C22);
                                end
                            end
                        end
                        C2=mean(X3,3);
                        D2=mean(X4,3);
                        apa=(C1-C2);
                        bet=2*(D1-D2);
                        naa=0;
                        for na=1:1
                            naa=naa+1;
                            opt_angle(:,:)=(-1/2)*acos(apa./(sqrt( (apa.*apa) + (bet.*bet) ))) + pi/4;
                        end
                    end
                    for naa=1:1
                        distance1(:,:)=C1.*sin(2*opt_angle(:,:))+D1.*cos(2*opt_angle(:,:));
                        distance2(:,:)=C2.*sin(2*opt_angle(:,:))+D2.*cos(2*opt_angle(:,:));
                        real_dis(:,:)=(distance1(:,:)-distance2(:,:)).^2;
                    end
                    for kk=1:siz
                        for k=kk+1:siz
                            [dist(kk,k),aaz(kk,k)]=max(real_dis(kk,k,:));
                        end
                    end
                    for kk=1:siz
                        for k=kk+1:siz
                            aa(kk,k)=opt_angle(kk,k);
                       end
                    end
                    aaf=aa*180/pi;
                    lp(:,:)=real_dis;
                    for trial=1:l
                        for k=1:siz
                            for kk=k+1:siz
                                cc_r=cov(n_ec1(trial,:,k),n_ec1(trial,:,kk));
                                aa2=(aa(k,kk));
                                ro=[cos(aa2) -sin(aa2);sin(aa2) cos(aa2)];
                                cc_rc1=ro'*cc_r*ro;
                                cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                                c_rr_e1(k,kk,trial)=cc_rr;
                            end
                        end
                    end
                    for trial=1:r
                        for k=1:siz
                            for kk=k+1:siz
                                cc_r2=cov(n_ec2(trial,:,k),n_ec2(trial,:,kk));
                                aa2=(aa(k,kk));
                                ro=[cos(aa2) -sin(aa2);sin(aa2) cos(aa2)];
                                cc_rc2=ro'*cc_r2*ro;
                                cc_rr2=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
                                c_rr_e2(k,kk,trial)=cc_rr2;
                            end
                        end
                    end
                    for trial=1:tr
                        for k=1:siz
                            for kk=k+1:siz
                                cc_r3=cov(n_ec3(trial,:,k),n_ec3(trial,:,kk));
                                aa2=(aa(k,kk));
                                ro=[cos(aa2) -sin(aa2);sin(aa2) cos(aa2)];
                                cc_rc3=ro'*cc_r3*ro;
                                cc_rr3=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
                                c_rr_e3(k,kk,trial)=cc_rr3;
                            end
                        end
                    end
                    fra=[];
                    kl=((mean(c_rr_e1,3)-mean(c_rr_e2,3)).^2);
                    oo=0;
                    for k=1:siz
                        for kk=k+1:siz
                            oo=oo+1;
                            fra(oo)=kl(k,kk);
                            fraa(oo)=aa(k,kk);
                        end
                    end
                    [ai1,ai2]=sort(fra,'ascend');
                    m_thr=length(find(fra>(mean(fra) )));
                    ww2=median(ai1(378-m_thr:333));
                    ww1=median(ai1(334:end));
                    iqr=ww1-ww2;
                    nma=ww1+1.5*(ww1-ww2);
                    point_set=length(find(fra>((nma) )))
                    nb=0;
                    for mp=point_set: point_set
                        nb=nb+1;
                        c_f1=[];c_f2=[];c_f3=[];
                        t_fl=sort(fra,'descend');
                        total_fl(1,:)=t_fl;
                        rp=kl;
                        rp(rp<t_fl(mp))=0;
                        rp(rp>0)=1;
                        V=rp;
%%
                        for trial=1:l
                            mask_1(:,:,trial)=c_rr_e1(:,:,trial).*V;
                            teg1=mask_1(:,:,trial);
                            qw=1;
                            qww=1;
                            for k=1:siz
                                for kk=k+1:siz
                                    tot1(1,qww)=c_rr_e1(k,kk,trial);
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
                            mask_2(:,:,trial)=c_rr_e2(:,:,trial).*V;
                            teg2=mask_2(:,:,trial);
                            qw=1;
                            qww=1;
                            for k=1:siz
                                for kk=k+1:siz
                                    tot2(1,qww)=c_rr_e2(k,kk,trial);
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
                            mask_3(:,:,trial)=c_rr_e3(:,:,trial).*V;
                            teg3=mask_3(:,:,trial);
                            qw=1;
                            qww=1;
                            for k=1:siz
                                for kk=k+1:siz
                                    tot3(1,qww)=c_rr_e3(k,kk,trial);
                                    qww=qww+1;
                                    if teg3(k,kk)~=0
                                        f3(qw)=teg3(k,kk);
                                        qw=qw+1;
                                    end
                                end
                            end
                            c_f3(trial,:)=f3;
                        end
                        vf=( [c_f1;c_f2] );
                        vft=( c_f3);
                        options.MaxIter = 100000;
                        lvf=[ones(l,1);ones(r,1)+1];
                        SVMStruct = svmtrain(vf,lvf,'Options', options);
                        mdl=fitcsvm(vf,lvf);
                        [result1,sco]= predict(mdl,vft);
                        result = svmclassify(SVMStruct,vft);
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
                        accy1(ccq,nb)=correct1/length(result1)*100;
                        accy(ccq,nb)=correct/length(result)*100;
                    end
                end
            end
        end
    end
end
