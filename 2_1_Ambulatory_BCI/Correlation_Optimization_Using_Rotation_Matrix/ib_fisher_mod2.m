clear all
load('Traindata_0.txt');
load('Traindata_1.txt');
load('Testdata.txt');
load('tl.txt');
% load('ttl.txt');
ttl=tl;
tvl=ttl'+1;
for tt=1:100
    for k=1:7
        eeg_1(k,:,tt)=Traindata_0(tt, (k-1)*1152+2:(k-1)*1152+1152);
    end
end
eege_1(1:4,:,:)=eeg_1(1:4,:,:);
eege_1(5:6,:,:)=eeg_1(6:7,:,:);
for k=1:100
    for kk=1:6
%                         e11(kk,:,k)=eege_1(kk,:,k)-sum(eege_1(:,:,k),1)/6;
        e11(kk,:,k)=eege_1(kk,:,k);
    end
end

for tt=1:100
    for k=1:7
        eeg_2(k,:,tt)=Traindata_1(tt, (k-1)*1152+2:(k-1)*1152+1152);
    end
end
eege_2(1:4,:,:)=eeg_2(1:4,:,:);
eege_2(5:6,:,:)=eeg_2(6:7,:,:);
for k=1:100
    for kk=1:6
%                         e22(kk,:,k)=eege_2(kk,:,k)-sum(eege_2(:,:,k),1)/6;
        e22(kk,:,k)=eege_2(kk,:,k);
    end
end
for tt=1:180
    for k=1:7
        eeg_3(k,:,tt)=Testdata(tt, (k-1)*1152+1:(k-1)*1152+1152);
    end
end
eege_3(1:4,:,:)=eeg_3(1:4,:,:);
eege_3(5:6,:,:)=eeg_3(6:7,:,:);
for k=1:180
    for kk=1:6
%                         e33(kk,:,k)=eege_3(kk,:,k)-sum(eege_3(:,:,k),1)/6;
        e33(kk,:,k)=eege_3(kk,:,k);
    end
end
% e11(1:4,:,:)=eg_1(1:4,:,:);
% e22(1:4,:,:)=eg_2(1:4,:,:);
% e33(1:4,:,:)=eg_3(1:4,:,:);
% e11(5:6,:,:)=eg_1(6:7,:,:);
% e22(5:6,:,:)=eg_2(6:7,:,:);
% e33(5:6,:,:)=eg_3(6:7,:,:);
l=100;
r=100;
tr=180;
%%
tsum1=[];
tsum2=[];
ccq=0;
for fqq=3:3%trs=0.6, 0.5~8.5, 6feature 88%
    fqq
    for bz=4:4
        for fq=0.6:0.05:0.6
            for nz=0.5:0.5:0.5
                ccq=ccq+1;
                [bbb,aaa]=butter(bz,[0.4/128 (7+nz)/128]);
                tend=500;
                f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];tf_ec1=[];tf_ec2=[];tf_ec3=[];
                n_tf_ec1=[];n_tf_ec2=[];n_tf_ec3=[];n_ec1=[];n_ec2=[];n_ec3=[];cc_rr_e1=[];ect1=[];ect2=[];ect3=[];
                cc_rr_e2=[];cc_rr_e3=[];cro1=[];cro2=[];cro3=[];c2ro1=[];c2ro2=[];c2ro3=[];mask_1=[];mask_2=[];mask_3=[];result=[];stemp=[];fe1=[];fe2=[];fe3=[];result1=[];stemp=[];sstemp=[];
                ssstemp=[];co1=[];co2=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];tlcsp=[];trcsp=[];ttcsp=[];V=[];Va=[];VV=[];
                for node=1:6
                    for k=1:l
                        ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
                        %             ect1(k,:)=e11(node,:,k);
                    end
                    n_ec1(:,:,node)=ect1(:,250+fqq:765);%(:,41:280
                    for k=1:r
                        ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
                        %             ect2(k,:)=e22(node,:,k);
                    end
                    n_ec2(:,:,node)=ect2(:,250+fqq:765);
                    for k=1:tr
                        ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
                        %             ect3(k,:)=e33(node,:,k);
                    end
                    n_ec3(:,:,node)=ect3(:,250+fqq:765);
                end
                
                
                %%
                for trial=1:l
                    for kk=1:6
                        for k=1:6
                            g1=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                            gg_rr=g1(2)/sqrt(g1(1,1)*g1(2,2));%(std(n_ec1(trial,:,kk))*std(n_ec1(trial,:,k)) );
                            gg_rr_e1(kk,k,trial)=gg_rr;
                        end
                    end
                end
                for trial=1:r
                    for kk=1:6
                        for k=1:6
                            g2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                            gg_rr2=g2(2)/sqrt(g2(1,1)*g2(2,2));%(std(n_ec1(trial,:,kk))*std(n_ec1(trial,:,k)) );
                            gg_rr_e2(kk,k,trial)=gg_rr2;
                        end
                    end
                end
                for trial=1:tr
                    for kk=1:6
                        for k=1:6
                            g3=cov(n_ec3(trial,:,kk),n_ec3(trial,:,k));
                            gg_rr3=g3(2)/sqrt(g3(1,1)*g3(2,2));%(std(n_ec1(trial,:,kk))*std(n_ec1(trial,:,k)) );
                            gg_rr_e3(kk,k,trial)=gg_rr3;
                        end
                    end
                end
                cor_check1=abs(mean(gg_rr_e1,3));
                cor_check2=abs(mean(gg_rr_e2,3));
                jk=0;
                for yo=1:6
                    for yyo=yo+1:6
                        jk=jk+1;
                        coq1(1,jk)=cor_check1(yo,yyo);
                        coq2(1,jk)=cor_check2(yo,yyo);
                    end
                end
                % fq=0.65;
                coq11=sort(coq1);
                coq22=sort(coq2);
                
                cor_check1(cor_check1>fq)=0;
                cor_check2(cor_check2>fq)=0;
                VV=cor_check1.*cor_check2;
                VV(VV>0)=1;
                for ch=1:6
                    [new_cl,new_cr,new_ct] =anc_ch6(l,r,tr,n_ec1,n_ec2,n_ec3,ch,VV,765-fqq-250+1);
                    n_ec1(:,:,ch)=new_cl;
                    n_ec2(:,:,ch)=new_cr;
                    n_ec3(:,:,ch)=new_ct;
                end
                %%
 ch=6
                    % zx=xxz;
                    ssi=0;
                    si=0;
                    aa=(si)*pi/180;
                    siz=6;
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
                                    C=sqrt( (cc_r(1,1)*cos(magle)*cos(magle)+cc_r(2,2)*sin(magle)*sin(magle)+cc_r(2)*sin(2*magle))*(cc_r(1,1)*sin(magle)*sin(magle)+cc_r(2,2)*cos(magle)*cos(magle)-cc_r(2)*sin(2*magle)));
                                    X1(kk,k,trial)=real(A/C);
                                    X2(kk,k,trial)=real(B/C);
                                end
                            end
                        end
                        C1=mean(X1,3);
                        D1=mean(X2,3);
                        for trial=1:l
                            for kk=1:siz
                                for k=kk+1:siz
                                    cc_r=cov(n_ec1(trial,:,kk),n_ec1(trial,:,k));
                                    A=(cc_r(2,2)-cc_r(1,1))/2;
                                    B=cc_r(2);
                                    magle=(pi/8)-(acos( cc_r(2)/ sqrt( cc_r(2)*cc_r(2)+ ( (cc_r(1,1)-cc_r(2,2))/2 )^2) ))/2;
                                    %                         magle=0;
                                    C=sqrt( (cc_r(1,1)*cos(magle)*cos(magle)+cc_r(2,2)*sin(magle)*sin(magle)+cc_r(2)*sin(2*magle))*(cc_r(1,1)*sin(magle)*sin(magle)+cc_r(2,2)*cos(magle)*cos(magle)-cc_r(2)*sin(2*magle)));
                                    xx1(kk,k,trial)=(real(A/C)-C1(kk,k)).^2;
                                    xx2(kk,k,trial)=(real(B/C)-D1(kk,k)).^2;
                                    y1(kk,k,trial)=sqrt(xx1(kk,k,trial).*xx1(kk,k,trial)+xx2(kk,k,trial).*xx2(kk,k,trial));
                                    cphi1(kk,k,trial)=cos(2*(  acos((xx1(kk,k,trial))./y1(kk,k,trial))) );
                                    sphi1(kk,k,trial)=sin(2*(  acos((xx1(kk,k,trial))./y1(kk,k,trial))) );
                                    yf1(kk,k,trial)=y1(kk,k,trial).*cphi1(kk,k,trial);
                                    yf2(kk,k,trial)=y1(kk,k,trial).*sphi1(kk,k,trial);
                                    %                         AA(kk,k,trial)=sqrt((y1(kk,k,trial)^4)*cos(2*phi1(kk,k,trial))*cos(2*phi1(kk,k,trial)) + (y1(kk,k,trial)^4)*sin(2*phi1(kk,k,trial))*cos(2*phi1(kk,k,trial)));
                                    %                         op_phi1(kk,k,trial)=acos( (y1(kk,k,trial)^2)*cos(2*phi1(kk,k,trial)) / AA(kk,k,trial));
                                end
                            end
                        end
                        
                        yc1=mean(yf1,3);
                        yc2=mean(yf2,3);
                        %%
                        for trial=1:r
                            for kk=1:siz
                                for k=kk+1:siz
                                    cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                                    A2=(cc_r2(2,2)-cc_r2(1,1));
                                    B2=2*cc_r2(2);
                                    
                                    magle=(pi/8)-(acos( cc_r2(2)/ sqrt( cc_r2(2)*cc_r2(2)+ ( (cc_r2(1,1)-cc_r2(2,2))/2 )^2) ))/2;
                                    %                         magle=0;
                                    C22=sqrt( (cc_r2(1,1)*cos(magle)*cos(magle)+cc_r2(2,2)*sin(magle)*sin(magle)+cc_r2(2)*sin(2*magle))*(cc_r2(1,1)*sin(magle)*sin(magle)+cc_r2(2,2)*cos(magle)*cos(magle)-cc_r2(2)*sin(2*magle)));
                                    X3(kk,k,trial)=real(A2/C22);
                                    X4(kk,k,trial)=real(B2/C22);
                                end
                            end
                        end
                        C2=mean(X3,3);
                        D2=mean(X4,3);
                        for trial=1:r
                            for kk=1:siz
                                for k=kk+1:siz
                                    cc_r2=cov(n_ec2(trial,:,kk),n_ec2(trial,:,k));
                                    A2=(cc_r2(2,2)-cc_r2(1,1))/2;
                                    B2=cc_r2(2);
                                    
                                    magle=(pi/8)-(acos( cc_r2(2)/ sqrt( cc_r2(2)*cc_r2(2)+ ( (cc_r2(1,1)-cc_r2(2,2))/2 )^2) ))/2;
                                    %                         magle=0;
                                    C22=sqrt( (cc_r2(1,1)*cos(magle)*cos(magle)+cc_r2(2,2)*sin(magle)*sin(magle)+cc_r2(2)*sin(2*magle))*(cc_r2(1,1)*sin(magle)*sin(magle)+cc_r2(2,2)*cos(magle)*cos(magle)-cc_r2(2)*sin(2*magle)));
                                    xx3(kk,k,trial)=(real(A2/C22)-C2(kk,k)).^2;
                                    xx4(kk,k,trial)=(real(B2/C22)-D2(kk,k)).^2;
                                    y2(kk,k,trial)=sqrt(xx3(kk,k,trial).*xx3(kk,k,trial)+xx4(kk,k,trial).*xx4(kk,k,trial));
                                    cphi2(kk,k,trial)=cos(2*(  acos((xx3(kk,k,trial))./y2(kk,k,trial))) );
                                    sphi2(kk,k,trial)=sin(2*(  acos((xx3(kk,k,trial))./y2(kk,k,trial))) );
                                    yyf1(kk,k,trial)=y2(kk,k,trial).*cphi2(kk,k,trial);
                                    yyf2(kk,k,trial)=y2(kk,k,trial).*sphi2(kk,k,trial);
                                end
                            end
                        end
                        yyc1=mean(yyf1,3);
                        yyc2=mean(yyf2,3);
                        %%
                        A_1=C1-C2;
                        B_1=D1-D2;
                        AA=A_1.*A_1 + B_1.*B_1;
                        PH1=acos(A_1./ sqrt(AA));
                        D_1=(mean(xx1,3).^2+mean(xx2,3).^2 +mean(xx3,3).^2 + mean(xx4,3).^2 )/2;
                        W_1=yc2+yyc2;
                        U_1=yc1+yyc1;
                        PH2=acos( -1*W_1 ./ sqrt(W_1.^2 + U_1.^2));
                        V_1=sqrt(W_1.^2 + U_1.^2);
                        AP=2*AA.*D_1.*cos(2*PH1)+(2.*AA.*V_1.*sin(PH2));
                        BP=2*AA.*D_1.*sin(2.*PH1)+(2.*AA.*V_1.*cos(PH2));
                        angle=((asin( (-2*AA.*V_1.*cos(2.*PH1-PH2)) ./ sqrt(AP.*AP+BP.*BP)) - acos( AP./sqrt(AP.*AP+BP.*BP) ) )/4);
                        %%
                        apa=2*(C1.*C1-C2.*C2);
                        bet=2*(D1-D2);
                        naa=0;
                        for na=-2:2
                            naa=naa+1;
                            %             opt_angle(:,:,naa)=(-1/4)*acos(apa./(2*sqrt( (apa.*apa./4) + (bet.*bet) ))) + na*pi/4;
                            % oopt_angle(:,:,naa)=-acos((yc1+yyc1)./ sqrt(   (yc1+yyc1).*(yc1+yyc1) + (yc2+yyc2).*(yc2+yyc2)) )/4 + na*pi/2;
                            oopt_angle(:,:,naa)=real((asin( (-2*AA.*V_1.*cos(2.*PH1-PH2)) ./ sqrt(AP.*AP+BP.*BP)) - acos( AP./sqrt(AP.*AP+BP.*BP) ) )+na*2*pi)/4;
%                             oopt_angle(:,:,naa)=real((asin( (-2*AA.*V_1.*cos(2.*-PH1+PH2)) ./ sqrt(AP.*AP+BP.*BP)) - acos( AP./sqrt(AP.*AP+BP.*BP) ) +na*2*pi)/4);
                        end
                        
                        %         for na=1:1
                        %             naa=naa+1;
                        % %             opt_angle(:,:,naa)=(-1/4)*acos(apa./(2*sqrt( (apa.*apa./4) + (bet.*bet) ))) + na*pi/4;
                        %             opt_angle(:,:)=(-1/2)*acos(apa./(sqrt( (apa.*apa) + (bet.*bet) ))) + pi/4;
                        % %              opt_angle(:,:,naa)=0;
                        %         end
                    end
                    %     naa=0;
                    for naa=1:5
                        %         naa=naa+1;
                        varian1(:,:,naa)=mean(xx1.^2,3).*sin(2*oopt_angle(:,:,naa)) + mean((xx1.*xx2),3).*sin(2*oopt_angle(:,:,naa)).*cos(2*oopt_angle(:,:,naa))+mean(xx2.^2,3).*cos(2*oopt_angle(:,:,naa));
                        varian2(:,:,naa)=mean(xx3.^2,3).*sin(2*oopt_angle(:,:,naa)) + mean((xx3.*xx4),3).*sin(2*oopt_angle(:,:,naa)).*cos(2*oopt_angle(:,:,naa))+mean(xx4.^2,3).*cos(2*oopt_angle(:,:,naa));
                        distance1(:,:,naa)=C1.*sin(2*oopt_angle(:,:,naa))+D1.*cos(2*oopt_angle(:,:,naa));
                        distance2(:,:,naa)=C2.*sin(2*oopt_angle(:,:,naa))+D2.*cos(2*oopt_angle(:,:,naa));
                        %         real_dis(:,:,naa)=(distance1(:,:,naa)-distance2(:,:,naa)).^2;
%                         real_dis(:,:,naa)=(AA.*sin( 2.*oopt_angle(:,:,naa)+PH1).^2)./ (D_1+V_1.*sin(4*oopt_angle(:,:,naa)+PH2));
                        real_dis(:,:,naa)=(AA.*sin( 2.*oopt_angle(:,:,naa)+PH1).^2)./(varian1(:,:,naa)+varian2(:,:,naa));
                    end
                    
                    for kk=1:siz
                        for k=kk+1:siz
                            [dist(kk,k),aaz(kk,k)]=max(real_dis(kk,k,:));
                        end
                    end
                    for kk=1:siz
                        for k=kk+1:siz
                            aa(kk,k)=oopt_angle(kk,k,aaz(kk,k));
                        end
                    end
                    aaf=aa*180/pi;
                    for kk=1:5
                        for k=1:6
                            if aaf(kk,k) < 0
                                aaf(kk,k)=aaf(kk,k)+90;
                            end
                            if aaf(kk,k) >90
                                aaf(kk,k)=aaf(kk,k)-90;
                            end
                        end
                    end
                    for kk=1:5
                        for k=1:6
                            if aaf(kk,k) < 0
                                aaf(kk,k)=aaf(kk,k)+90;
                            end
                            if aaf(kk,k) >90
                                aaf(kk,k)=aaf(kk,k)-90;
                            end
                        end
                    end
                    for kk=1:5
                        for k=1:6
                            if aaf(kk,k) < 0
                                aaf(kk,k)=aaf(kk,k)+90;
                            end
                            if aaf(kk,k) >90
                                aaf(kk,k)=aaf(kk,k)-90;
                            end
                        end
                    end
                    for kk=1:5
                        for k=1:6
                            if aaf(kk,k) < 0
                                aaf(kk,k)=aaf(kk,k)+90;
                            end
                            if aaf(kk,k) >90
                                aaf(kk,k)=aaf(kk,k)-90;
                            end
                        end
                    end
                aa=aaf*pi/180;
%                 aa=aaf*0;
                lp(:,:)=dist;
                for trial=1:l
                    for k=1:siz
                        for kk=k+1:siz
                            cc_r=cov(n_ec1(trial,:,k),n_ec1(trial,:,kk));
                            aa2=(aa(k,kk));
                            ro=[cos(aa2) -sin(aa2);sin(aa2) cos(aa2)];
                            cc_rc1=ro'*cc_r*ro;
                            cc_rr=cc_rc1(2)/sqrt(cc_rc1(1,1)*cc_rc1(2,2));%(std(n_ec1(trial,:,kk))*std(n_ec1(trial,:,k)) );
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
                % for k=1:siz
                %     for kk=k+1:siz
                %         lp(k,kk)=0;
                %     end
                % end
                 kl=((mean(c_rr_e1,3)-mean(c_rr_e2,3)).^2)./((var(c_rr_e1,0,3)+var(c_rr_e2,0,3)));
                 kl=((mean(c_rr_e1,3)-mean(c_rr_e2,3)).^2);
                oo=0;
                for k=1:siz
                    for kk=k+1:siz
                        oo=oo+1;
                        fra(oo)=kl(k,kk);
                        fraa(oo)=aa(k,kk);
                    end
                end
                point_set=length(find(fra>mean(fra)))
                nb=0;
                for mp=point_set:point_set
                    nb=nb+1;
                    c_f1=[];c_f2=[];c_f3=[];
                    t_fl=sort(fra,'descend');
                    total_fl(1,:)=t_fl;
                    rp=kl;
                    rp(rp<t_fl(mp))=0;
                    %     rp=lp;
                    rp(rp>0)=1;
                    V=rp;
                    
                    %%
                    %     toto(ccq)=sum(sum(V))
                    
                    for trial=1:l
                        mask_1(:,:,trial)=c_rr_e1(:,:,trial).*V;
                        teg1=mask_1(:,:,trial);
                        qw=1;
                        qww=1;
                        for k=1:siz
                            for kk=k+1:siz
                                tot1(1,qww)=c_rr_e1(k,kk,trial);
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
                        mask_2(:,:,trial)=c_rr_e2(:,:,trial).*V;
                        
                        teg2=mask_2(:,:,trial);
                        qw=1;
                        qww=1;
                        for k=1:siz
                            for kk=k+1:siz
                                tot2(1,qww)=c_rr_e2(k,kk,trial);
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
                        mask_3(:,:,trial)=c_rr_e3(:,:,trial).*V;
                        teg3=mask_3(:,:,trial);
                        qw=1;
                        qww=1;
                        for k=1:siz
                            for kk=k+1:siz
                                tot3(1,qww)=c_rr_e3(k,kk,trial);
                                qww=qww+1;
                                if teg3(k,kk)~=0;
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
                    
                    %     vf=([lef;rig]);
                    %     vft=res;
                    SVMStruct = svmtrain(vf,lvf,'Options', options);
                    mdl=fitcsvm(vf,lvf);
                    %      mdll=compact(mdl);
                    %         tvl=true_y((trn+1):end);
                    %      mdll = mdl.Trained{1}
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