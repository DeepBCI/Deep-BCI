clear all
load('Traindata_0.txt');
load('Traindata_1.txt');
load('Testdata.txt');
load('tl.txt');
tvl=tl'+1;
for tt=1:100
    for k=1:7
        eeg_1(k,:,tt)=Traindata_0(tt, (k-1)*1152+2:(k-1)*1152+1152);
    end
end
for k=1:100
    for kk=1:7
        eg_1(kk,:,k)=eeg_1(kk,:,k);
    end
end
for tt=1:100
    for k=1:7
        eeg_2(k,:,tt)=Traindata_1(tt, (k-1)*1152+2:(k-1)*1152+1152);
    end
end
for k=1:100
    for kk=1:7
        eg_2(kk,:,k)=eeg_2(kk,:,k);
    end
end
for tt=1:180
    for k=1:7
        eeg_3(k,:,tt)=Testdata(tt, (k-1)*1152+1:(k-1)*1152+1152);
    end
end
for k=1:180
    for kk=1:7
        eg_3(kk,:,k)=eeg_3(kk,:,k);
    end
end
e11=eg_1(1:4,:,:);
e22=eg_2(1:4,:,:);
e33=eg_3(1:4,:,:);
e11(5:6,:,:)=eg_1(6:7,:,:);
e22(5:6,:,:)=eg_2(6:7,:,:);
e33(5:6,:,:)=eg_3(6:7,:,:);

l=100;
r=100;
tr=180;
%%

ccq=0;
for fegg=0.5:0.5:0.5 %0.5-8/10 features : 5th butter : 63.8889
    for feg=8:0.5:8
        [bbb,aaa]=butter(5,[fegg/128 feg/128]);
        f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];n_ec1=[];n_ec2=[];n_ec3=[];cc_rr_e1=[];ect1=[];ect2=[];ect3=[];
        cc_rr_e2=[];cc_rr_e3=[];mask_1=[];mask_2=[];mask_3=[];result=[];result1=[];V=[];VV=[];
        for ths=0.7:0.05:0.7  %reference signal correlation
            ccq=ccq+1;
            for node=1:6
                for k=1:l
                    ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
                end
                n_ec1(:,:,node)=ect1(:,256:768);
                for k=1:r
                    ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
                 end
                n_ec2(:,:,node)=ect2(:,256:768);
                for k=1:tr
                    ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
                  end
                n_ec3(:,:,node)=ect3(:,256:768);
            end
            
            aa=(0)*pi/180;
            ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
            for trial=1:l
                for kk=1:6
                    for k=1:6
                        cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                        cc_rc1=ro'*cc_r*ro;
                        cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                        cc_rr_e1(kk,k,trial)=cc_rr;
                    end
                end
            end
            for trial=1:r
                for kk=1:6
                    for k=1:6
                        cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                        cc_rc2=ro'*cc_r2*ro;
                        cc_rr2=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
                        cc_rr_e2(kk,k,trial)=cc_rr2;
                    end
                end
            end
            for trial=1:tr
                for kk=1:6
                    for k=1:6
                        cc_r3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
                        cc_rc3=ro'*cc_r3*ro;
                        cc_rr3=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
                        cc_rr_e3(kk,k,trial)=cc_rr3;
                    end
                end
            end
            test_c=cc_rr_e1(:,:,1);
            cor_check1=abs(mean(cc_rr_e1,3));
            cor_check2=abs(mean(cc_rr_e2,3));
            jk=0;
            for yo=1:6
                for yyo=yo+1:6
                    jk=jk+1;
                    coq1(1,jk)=cor_check1(yo,yyo);
                    coq2(1,jk)=cor_check2(yo,yyo);
                end
            end
            coq11=sort(coq1);
            coq22=sort(coq2);
            cor_check1(cor_check1>ths)=0;
            cor_check2(cor_check2>ths)=0;
            VV=cor_check1.*cor_check2;
            VV(VV>0)=1;
            
%% adaptive noise cancellaion
            for ch=1:6
                [new_cl,new_cr,new_ct] =anc_ch6(l,r,tr,n_ec1,n_ec2,n_ec3,ch,VV,513);
                n_ec1(:,:,ch)=new_cl;
                n_ec2(:,:,ch)=new_cr;
                n_ec3(:,:,ch)=new_ct;
            end
%% rotatation matrix
            cq=0;
            for si=-30:5:30 %angle
                cq=cq+1;
                si
                aa=(si)*pi/180;
                ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
                for trial=1:l
                    for kk=1:6
                        for k=1:6
                            cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                            cc_rc1=ro'*cc_r*ro;
                            cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                            cc_rr_e1(kk,k,trial)=cc_rr;
                        end
                    end
                end
                for trial=1:r
                    for kk=1:6
                        for k=1:6
                            cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                            cc_rc2=ro'*cc_r2*ro;
                            cc_rr2=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
                            cc_rr_e2(kk,k,trial)=cc_rr2;
                        end
                    end
                end
                for trial=1:tr
                    for kk=1:6
                        for k=1:6
                            cc_r3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
                            cc_rc3=ro'*cc_r3*ro;
                            cc_rr3=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
                            cc_rr_e3(kk,k,trial)=cc_rr3;
                        end
                    end
                end
                %% fisher ratio based ratated angle selection
                m_ce1=sum( cc_rr_e1,3)/l;
                m_ce2=sum( cc_rr_e2,3)/r;
                ui=abs(m_ce1-m_ce2);
                kj=m_ce1-m_ce2;
                st1=std(cc_rr_e1,0,3);
                st2=std(cc_rr_e2,0,3);
                va1=var(cc_rr_e1,0,3);
                va2=var(cc_rr_e2,0,3);
                tss=(kj)./sqrt( (st1.^2/l+st2.^2/r) );
                fratio(:,:,cq)=(kj).^2./(va1+va2) ;
                for k=1:6
                    fratio(k,k,cq)=1;
                end
            end
            for k=1:6
                for kk=1:6
                    [lp(k,kk),llp(k,kk)]=max(fratio(k,kk,:));
                end
            end
            %%
            llp=llp-31;
            llp=5*(llp+30)-30;
            hn=0;
            for k=1:6
                for kk=k+1:6
                    hn=hn+1;
                    sort_angle(hn)=llp(k,kk);
                end
            end
            for trial=1:l
                for k=1:6
                    for kk=1:6
                        cc_r=cov(n_ec1(trial,:,k),n_ec1(trial,:,kk));
                        aa=(llp(k,kk))*pi/180;
                        ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
                        cc_rc1=ro'*cc_r*ro;
                        cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));
                        cc_rr_e1(kk,k,trial)=cc_rr;
                    end
                end
            end
            for trial=1:r
                for k=1:6
                    for kk=1:6
                        cc_r2=cov(n_ec2(trial,:,k),n_ec2(trial,:,kk));
                        aa=(llp(k,kk))*pi/180;
                        ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
                        cc_rc2=ro'*cc_r2*ro;
                        cc_rr2=cc_rc2(2)/sqrt(cc_rc2(1,1)*cc_rc2(2,2));
                        cc_rr_e2(kk,k,trial)=cc_rr2;
                    end
                end
            end
            for trial=1:tr
                for k=1:6
                    for kk=1:6
                        cc_r3=cov(n_ec3(trial,:,k),n_ec3(trial,:,kk));
                        aa=(llp(k,kk))*pi/180;
                        ro=[cos(aa) -sin(aa);sin(aa) cos(aa)];
                        cc_rc3=ro'*cc_r3*ro;
                        cc_rr3=cc_rc3(2)/sqrt(cc_rc3(1,1)*cc_rc3(2,2));
                        cc_rr_e3(kk,k,trial)=cc_rr3;
                    end
                end
            end
            fra=[];
            for k=1:6
                for kk=k+1:6
                    lp(kk,k)=0;
                end
            end
            oo=0;
            for k=1:6
                for kk=k+1:6
                    oo=oo+1;
                    fra(oo)=lp(k,kk);
                    fraa(oo)=llp(k,kk);
                end
            end
            
            nb=0;
            for mp=10:10
                nb=nb+1;
                c_f1=[];c_f2=[];c_f3=[];
                [bn,bnn]=sort(fra,'descend');
                t_fl=sort(fra,'descend');
                rp=lp;
                rp(rp<t_fl(mp))=0;
                %     rp=lp;
                rp(rp>0)=1;
                V=rp;
                for trial=1:l
                    mask_1(:,:,trial)=cc_rr_e1(:,:,trial).*V;
                    teg1=mask_1(:,:,trial);
                    qw=1;
                    qww=1;
                    for k=1:6
                        for kk=k+1:6
                            tot1(1,qww)=cc_rr_e1(k,kk,trial);
                            qww=qww+1;
                            if teg1(k,kk)~=0;
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
                    for k=1:6
                        for kk=k+1:6
                            tot2(1,qww)=cc_rr_e2(k,kk,trial);
                            qww=qww+1;
                            if teg1(k,kk)~=0;
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
                    for k=1:6
                        for kk=k+1:6
                            tot3(1,qww)=cc_rr_e3(k,kk,trial);
                            qww=qww+1;
                            if teg3(k,kk)~=0;
                                f3(qw)=teg3(k,kk);
                                qw=qw+1;
                            end
                        end
                    end
                    c_f3(trial,:)=f3;
                    
                end
                %%
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
                lf=zeros(l,1);
                rf=zeros(r,1)+1;
                tf=tvl'-1;
                cc1=[lf,c_f1];
                cc2=[rf,c_f2];
                cc3=[tf,c_f3];
                cc4=[cc1;cc2];
                [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] =elm2(cc4,cc3, 1, 20, 'sig');
                accy1(ccq,nb)=correct1/length(result1)*100;
                accy(ccq,nb)=correct/length(result)*100;
                accy2(ccq,nb)=TestingAccuracy;
            end
        end
    end
end