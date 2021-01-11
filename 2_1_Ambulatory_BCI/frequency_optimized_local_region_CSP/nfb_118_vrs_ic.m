clear all %al,aa,av : standard
load('data_set_IVa_ay');
load('true_labels_ay');
load('tchannel');
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
numt=28;
for k=1:numt
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end
u=1;
for k=numt+1:280
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    ee3(:,:,u)=temp;
    u=u+1;
    temp=0;
end
st=1;
stt=1;
for k=1:numt
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
tr=280-numt;
for k=1:l
    ee1(:,:,k)=eeg(:,:,ll(k));
end
for k=1:r
    ee2(:,:,k)=eeg(:,:,rr(k));
end
n_ch=length(tchannl);
chn=[1:118];
for k=1:118
    e11(k,:,:)=ee1(chn(k),:,:);
    e22(k,:,:)=ee2(chn(k),:,:);
    e33(k,:,:)=ee3(chn(k),:,:);
end
%%
rre{1,1}=[1,2,6,7,10];
rre{1,2}=[1,2,3,7,11];
rre{1,3}=[2,3,4,11,12];
rre{1,4}=[3,4,5,8,12];
rre{1,5}=[4,5,8,9,13];
rre{1,6}=[1,6,7,10,14,15];
rre{1,7}=[1,2,6,7,10,11,15,16,17];
rre{1,8}=[4,5,9,8,12,13,19,20,21];
rre{1,9}=[5,8,9,13,21,22];
rre{1,10}=[6,7,10,14,15,16];
rre{1,11}=[7,11,16,17,18];
rre{1,12}=[8,12,18,19,20];
rre{1,13}=[8,9,13,20,21,22];
rre{1,14}=[6,10,14,15,23,31,32];
rre{1,15}=[6,7,10,14,15,16,23,24,33];
rre{1,16}=[7,10,11,15,16,17,24,25,34];
rre{1,17}=[7,11,16,17,18,25,26,35];
rre{1,18}=[11,12,17,18,19,26,27,36];
rre{1,19}=[8,12,18,19,20,27,28,37];
rre{1,20}=[8,12,13,19,20,21,28,29,38];
rre{1,21}=[8,9,13,20,21,22,29,30,39];
rre{1,22}=[9,13,21,22,30,40,41];
rre{1,23}=[14,15,23,24,32,33,31];
rre{1,24}=[15,16,23,24,25,33,34,43];
rre{1,25}=[16,17,24,25,26,34,35,44];
rre{1,26}=[17,18,25,26,27,35,36,45];
rre{1,27}=[18,19,26,27,28,36,37,46];
rre{1,28}=[19,20,27,28,29,37,38,47];
rre{1,29}=[20,21,28,29,30,38,39,48];
rre{1,30}=[21,22,29,30,39,40,41];
rre{1,31}=[14,31,32,50];
rre{1,32}=[14,23,31,32,33,42,50];
rre{1,33}=[15,23,24,32,33,34,42,43,51];
rre{1,34}=[16,24,25,33,34,35,43,44,52];
rre{1,35}=[15,23,24,32,33,34,42,43,51]+2;
rre{1,36}=[15,23,24,32,33,34,42,43,51]+3;
rre{1,37}=[15,23,24,32,33,34,42,43,51]+4;
rre{1,38}=[15,23,24,32,33,34,42,43,51]+5;
rre{1,39}=[15,23,24,32,33,34,42,43,51]+6;
rre{1,40}=[22,30,39,49,58,40,41];
rre{1,41}=[22,40,41,58];
rre{1,42}=[23,32,33,42,43,50,51,59];
rre{1,43}=[24,33,34,42,43,44,51,52,60];
rre{1,44}=[24,33,34,42,43,44,51,52,60]+1;
rre{1,45}=[24,33,34,42,43,44,51,52,60]+2;
rre{1,46}=[24,33,34,42,43,44,51,52,60]+3;
rre{1,47}=[24,33,34,42,43,44,51,52,60]+4;
rre{1,48}=[24,33,34,42,43,44,51,52,60]+5;
rre{1,49}=[30,39,40,48,49,57,58,66];
rre{1,50}=[31,32,42,50,51,59,67,68];
rre{1,51}=[33,42,43,50,51,52,59,60,69];
rre{1,52}=[33,42,43,50,51,52,59,60,69]+1;
rre{1,53}=[33,42,43,50,51,52,59,60,69]+2;
rre{1,54}=[33,42,43,50,51,52,59,60,69]+3;
rre{1,55}=[33,42,43,50,51,52,59,60,69]+4;
rre{1,56}=[33,42,43,50,51,52,59,60,69]+5;
rre{1,57}=[33,42,43,50,51,52,59,60,69]+6;
rre{1,58}=[40,41,49,57,58,66,76,77];
rre{1,59}=[42,50,51,59,60,68,69,78];
rre{1,60}=[43,51,52,59,60,61,69,70,79];
rre{1,61}=[43,51,52,59,60,61,69,70,79]+1;
rre{1,62}=[43,51,52,59,60,61,69,70,79]+2;
rre{1,63}=[43,51,52,59,60,61,69,70,79]+3;
rre{1,64}=[43,51,52,59,60,61,69,70,79]+4;
rre{1,65}=[43,51,52,59,60,61,69,70,79]+5;
rre{1,66}=[49,57,58,65,66,75,76,85];
rre{1,67}=[50,67,68,86,87];
rre{1,68}=[50,59,67,68,69,78,86,87];
rre{1,69}=[51,59,60,68,69,70,78,79,88];
rre{1,70}=[51,59,60,68,69,70,78,79,88]+1;
rre{1,71}=[51,59,60,68,69,70,78,79,88]+2;
rre{1,72}=[51,59,60,68,69,70,78,79,88]+3;
rre{1,73}=[51,59,60,68,69,70,78,79,88]+4;
rre{1,74}=[51,59,60,68,69,70,78,79,88]+5;
rre{1,75}=[51,59,60,68,69,70,78,79,88]+6;
rre{1,76}=[58,66,75,76,77,85,95,96];
rre{1,77}=[58,76,77,95,96];
rre{1,78}=[59,68,69,78,79,87,88,97];
rre{1,79}=[60,69,70,78,79,80,88,89,98];
rre{1,80}=[61,70,71,79,80,81,89,90];
rre{1,81}=[62,71,72,80,81,82,90,91];
rre{1,82}=[63,72,73,81,82,83,91,92];
rre{1,83}=[63,72,73,81,82,83,91,92]+1;
rre{1,84}=[63,72,73,81,82,83,91,92]+2;
rre{1,85}=[66,75,76,84,85,94,95,102];
rre{1,86}=[67,68,86,87,97,103];
rre{1,87}=[67,68,78,86,87,88,97,98];
rre{1,88}=[69,78,79,87,88,89,97,98,103];
rre{1,89}=[70,79,80,88,89,90,98,99,104];
rre{1,90}=[71,80,81,89,90,91,104,99];
rre{1,91}=[72,81,82,90,91,92,99,100,106];
rre{1,92}=[73,82,83,91,92,93,100,108];
rre{1,93}=[74,83,84,92,93,94,100,108,101];
rre{1,94}=[78,84,85,93,94,95,101,102,108];
rre{1,95}=[76,77,85,94,95,96,101,102];
rre{1,96}=[76,77,95,96,102,109];
rre{1,97}=[86,87,88,97,98,103];
rre{1,98}=[87,88,89,97,98,103,104,112];
rre{1,99}=[89,90,91,99,104,105,106];
rre{1,100}=[91,92,93,100,106,107,108];
rre{1,101}=[93,94,95,101,102,108,109,114];
rre{1,102}=[94,95,96,101,102,109];
rre{1,103}=[97,98,103,104,112];
rre{1,104}=[89,98,99,103,104,105,110,112];
rre{1,105}=[99,104,105,106,110];
rre{1,106}=[91,99,100,105,106,107,110,111];
rre{1,107}=[100,106,107,108,111];
rre{1,108}=[93,100,101,107,108,109,111,114];
rre{1,109}=[101,102,108,109,114];
rre{1,110}=[104,105,106,110,112,113,115];
rre{1,111}=[106,107,108,111,114,113,116];
rre{1,112}=[98,103,104,110,112,115,117];
rre{1,113}=[106,110,111,113,115,116];
rre{1,114}=[101,108,111,114,109,116,118];
rre{1,115}=[110,112,113,115,117];
rre{1,116}=[111,113,116,114,118];
rre{1,117}=[112,115,113,117];
rre{1,118}=[113,116,114,118];

%%
nr=size(rre,2);

ji=0;
chs=0;
for si=4:4:4
    chs=chs+1;
    fsz=4;
    ji=ji+1;
    jji=0;
    for ssi=si+4:si+4
        jji=jji+1;
        [bbb,aaa]=butter(4,[7/50 30/50]); %[9 30] cht=7 76.79
        [bbb1,aaa1]=butter(4,[16/50 24/50],'stop');
        f1=[];c_f1=[];f2=[];c_f2=[];f3=[];c_f3=[];vf=[];vft=[];tf_ec1=[];tf_ec2=[];tf_ec3=[];
        n_tf_ec1=[];n_tf_ec2=[];n_tf_ec3=[];n_ec1=[];n_ec2=[];n_ec3=[];ect1=[];ect2=[];ect3=[];
        cro1=[];cro2=[];cro3=[];c2ro1=[];c2ro2=[];c2ro3=[];mask_1=[];mask_2=[];mask_3=[];result=[];stemp=[];fe1=[];fe2=[];fe3=[];result1=[];stemp=[];sstemp=[];
        ssstemp=[];co1=[];co2=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];tlcsp=[];trcsp=[];ttcsp=[];V=[];Va=[];VV=[];
          
        for node=1:118
            for k=1:l
                ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
            end
            n_ec1_{ji,jji}(:,:,node)=(ect1(:,50:300));
            for k=1:r
                ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
            end
            n_ec2_{ji,jji}(:,:,node)=(ect2(:,50:300));
            for k=1:tr
                ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
            end
            n_ec3_{ji,jji}(:,:,node)=(ect3(:,50:300));
        end
    end
    %% re1
    for ua=1:nr
        r1=[];
        r1=rre{ua};
        for op=chs:chs
            for oop=1:jji
                [re_l_{op,oop,ua},re_r_{op,oop,ua},re_t_{op,oop,ua},re_d_{op,oop,ua}]=ppyk1(l,r,tr,n_ec1_{op,oop}(:,:,r1),n_ec2_{op,oop}(:,:,r1),n_ec3_{op,oop}(:,:,r1));
            end
        end
        
        [dr1{ua},ddr1{ua}]=sel5_aw(re_l_,re_r_,numt,chs,jji,l,r,ua);
        [fl{ua},fr{ua},ft{ua},do{ua},fs{ua}]=fea_sel4(re_l_,re_r_,re_t_,re_d_,ddr1{ua},jji,chs,ua);
        mi{ua}(chs,:)=dr1{ua};
        mii{ua}(chs,:)=ddr1{ua};
        lcsp{ua}(:,:,chs)=[fl{ua}];
        rcsp{ua}(:,:,chs)=[fr{ua}];
        tcsp{ua}(:,:,chs)=[ft{ua}];
        dif_t{ua}(chs,:)=[do{ua}];
    end
end
%%
for k=1:nr
    ddo{k}=dif_t{1,k};
    fffl{k}=lcsp{1,k};
    fffr{k}=rcsp{1,k};
    ffft{k}=tcsp{1,k};
end
for k=1:nr
    eig_d(k)=ddo{k};
end
% point_set=length(find(eig_d>(max(eig_d)-min(eig_d) )))
point_set=length(find(eig_d>=mean(eig_d)))
[fi1,fi2]=sort(eig_d,'descend');
ww1=median(fi1(1:58));
ww2=median(fi1(61:end));
iqr=ww1-ww2;
zx=0;
for mo=1.5:0.1:1.5
    sww1=[];
    sww2=[];
    sww=[];sdd=[];sw1=[];sw2=[];sw=[];sd=[];sav=[];sav1=[];
    zx=zx+1;
    nma=ww2+mo*(ww1-ww2);
    point_set=length(find(eig_d>((nma) )))
    for k=1:nr
        f_fl(:,(2*k-1):2*k)=fffl{fi2(k)};
        f_fr(:,(2*k-1):2*k)=fffr{fi2(k)};
        f_ft(:,(2*k-1):2*k)=ffft{fi2(k)};
    end
    m_f_fl=mean(f_fl);
    dfl=(f_fl-m_f_fl).^2;
    m_f_fr=mean(f_fr);
    dfr=(f_fr-m_f_fr).^2;
    qq=0;
    for k=2:2:(nr*2)
        qq=qq+1;
        sww1(qq)=((sum(sum(dfl(:,1:k)))/l)/k);
        sww2(qq)=((sum(sum(dfr(:,1:k)))/r)/k);
        sww(qq)=(sww1(qq)+sww2(qq))/2;
        sdd(qq)=sum((m_f_fl(1,1:k)-m_f_fr(1,1:k)).^2)/k;
    end
    pp=0;
    for p=point_set:point_set
        clcsp=[];crcsp=[];ctcsp=[];go1=[];sav=[];trs=0;
        pp=pp+1;
        trs=sww(p)/sdd(p);
        trs1=sdd(p);
        qq=0;
        for k=1:2:((point_set*2)-1)
            qq=qq+1;
            sw1(qq)=((sum(sum(dfl(:,k:k+1)))/l)/2);
            sw2(qq)=((sum(sum(dfr(:,k:k+1)))/r)/2);
            sw(qq)=(sw1(qq)+sw2(qq))/2;
            %         sd(qq)=sum((m_f_fl(1,k:k+1)-m_f_fr(1,k:k+1)).^2)/2;
            sd(qq)=norm((m_f_fl(1,k:k+3)-m_f_fr(1,k:k+3)));
        end
        go1=(sw./sd)-trs;
        go2=mean(sd)-sd;
        ii=0;
        for k=1:point_set
            if go2(k)<=0
                ii=ii+1;
                sav(ii)=k;
            end
        end
        i1=0;
        go3=go2(1:point_set);
        for k=1:point_set
            if go3(k)<=0
                i1=i1+1;
                sav1(i1)=k;
            end
        end
        for kkk=1:length(sav1)
            clcsp(:,(2*kkk-1):2*kkk)=fffl{fi2(sav1(kkk))};
            crcsp(:,(2*kkk-1):2*kkk)=fffr{fi2(sav1(kkk))};
            ctcsp(:,(2*kkk-1):2*kkk)=ffft{fi2(sav1(kkk))};
        end
        vf=[clcsp;crcsp];
        lvf=[ones(l,1);ones(r,1)+1];
        lvf=[ones(l,1);ones(r,1)+1];
        options.MaxIter = 100000;
        SVMStruct = svmtrain(vf,lvf,'Options', options);
        mdl=fitcsvm(vf,lvf);%,'BoxConstraint',0.26097,'KernelScale',0.021833);
        %     mdl=fitcsvm(vf,lvf,'Standardize',true);
        tvl=true_y(numt+1:end);
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
        
        aaccy1(zx)=correct1/length(result1)*100;
        aaccy(zx)=correct/length(result)*100;
        aca(zx)=length(sav1);
    end
end