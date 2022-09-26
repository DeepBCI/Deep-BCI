clear all
clc

load('data_set_IVa_av');
load('true_labels_av');
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
for k=1:84
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    eeg(:,:,k)=temp;
    temp=0;
end
u=1;
for k=85:280
    temp=cnt(cue(k):cue(k)+500,:);
    temp=temp';
    ee3(:,:,u)=temp;
    u=u+1;
    temp=0;
end
st=1;
stt=1;
for k=1:84
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
tr=280-84;
for k=1:l
    ee1(:,:,k)=eeg(:,:,ll(k));
end
for k=1:r
    ee2(:,:,k)=eeg(:,:,rr(k));
end
chn=[50,43,44,52,53,60,61,89,54,91,55,47,48,56,58,64,65,93];%used channel 118channel -> 18channel
for k=1:18
    e11(k,:,:)=ee1(chn(k),:,:);
    e22(k,:,:)=ee2(chn(k),:,:);
    e33(k,:,:)=ee3(chn(k),:,:);
end
ji=0;
chs=0;
for si=4:4 :32 % # of filter banks
    chs=chs+1;
    fsz=4;
    ji=ji+1;
    jji=0;
    for ssi=si+4:si+4
        jji=jji+1;
        [bbb,aaa]=butter(4,[si/50 ssi/50]); 
        vf=[];vft=[];n_ec1=[];n_ec2=[];n_ec3=[];ect1=[];ect2=[];ect3=[];result=[];result1=[];     
        for node=1:18
            for k=1:l
                ect1(k,:)=filtfilt(bbb,aaa,e11(node,:,k));
            end
            n_ec1_{ji,jji}(:,:,node)=(ect1(:,51:300));
            for k=1:r
                ect2(k,:)=filtfilt(bbb,aaa,e22(node,:,k));
            end
            n_ec2_{ji,jji}(:,:,node)=(ect2(:,51:300));
            for k=1:tr
                ect3(k,:)=filtfilt(bbb,aaa,e33(node,:,k));
            end
            n_ec3_{ji,jji}(:,:,node)=(ect3(:,51:300));
        end
    end
    %% re1
    r1=[1,2,4,6];
    for op=chs:chs
        for oop=1:jji % FBCSP
            [re1_l_{op,oop},re1_r_{op,oop},re1_t_{op,oop},re1_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r1),n_ec2_{op,oop}(:,:,r1),n_ec3_{op,oop}(:,:,r1));
        end
    end
    [dr1,ddr1]=sel2_av2(re1_l_,re1_r_,84,chs,jji,l,r); %Mutual informaion
    [fl1,fr1,ft1,do1]=fea_sel3(re1_l_,re1_r_,re1_t_,re1_d_,ddr1,jji,chs); %MIBIF
    mi1(chs,:)=dr1;
    mii1(chs,:)=ddr1;
    lcsp1(:,:,chs)=[fl1];
    rcsp1(:,:,chs)=[fr1];
    tcsp1(:,:,chs)=[ft1];
    dif_t1(chs,:)=[do1]; % Local region discrimiation based on eigenvalue
    %% re2
    r2=[1,2,3,4];
    for op=chs:chs
        for oop=1:jji
            [re2_l_{op,oop},re2_r_{op,oop},re2_t_{op,oop},re2_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r2),n_ec2_{op,oop}(:,:,r2),n_ec3_{op,oop}(:,:,r2));
        end
    end
    [dr2,ddr2]=sel2_av2(re2_l_,re2_r_,84,chs,jji,l,r);
    [fl2,fr2,ft2,do2]=fea_sel3(re2_l_,re2_r_,re2_t_,re2_d_,ddr2,jji,chs);
    mi2(chs,:)=dr2;
    mii2(chs,:)=ddr2;
    lcsp2(:,:,chs)=[fl2];
    rcsp2(:,:,chs)=[fr2];
    tcsp2(:,:,chs)=[ft2];
    dif_t2(chs,:)=[do2];
    %% re3
    r3=[2,3,4,5,9];
    for op=chs:chs
        for oop=1:jji
            [re3_l_{op,oop},re3_r_{op,oop},re3_t_{op,oop},re3_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r3),n_ec2_{op,oop}(:,:,r3),n_ec3_{op,oop}(:,:,r3));
        end
    end
    [dr3,ddr3]=sel2_av2(re3_l_,re3_r_,84,chs,jji,l,r);
    [fl3,fr3,ft3,do3]=fea_sel3(re3_l_,re3_r_,re3_t_,re3_d_,ddr3,jji,chs);
    mi3(chs,:)=dr3;
    mii3(chs,:)=ddr3;
    lcsp3(:,:,chs)=[fl3];
    rcsp3(:,:,chs)=[fr3];
    tcsp3(:,:,chs)=[ft3];
    dif_t3(chs,:)=[do3];
    %% re4
    r4=[1,2,3,4,5,6,7];
    for op=chs:chs
        for oop=1:jji
            [re4_l_{op,oop},re4_r_{op,oop},re4_t_{op,oop},re4_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r4),n_ec2_{op,oop}(:,:,r4),n_ec3_{op,oop}(:,:,r4));
        end
    end
    [dr4,ddr4]=sel2_av2(re4_l_,re4_r_,84,chs,jji,l,r);
    [fl4,fr4,ft4,do4]=fea_sel3(re4_l_,re4_r_,re4_t_,re4_d_,ddr4,jji,chs);
    mi4(chs,:)=dr4;
    mii4(chs,:)=ddr4;
    lcsp4(:,:,chs)=[fl4];
    rcsp4(:,:,chs)=[fr4];
    tcsp4(:,:,chs)=[ft4];
    dif_t4(chs,:)=[do4];
    %% re5
    r5=[3,4,5,7,9];
    for op=chs:chs
        for oop=1:jji
            [re5_l_{op,oop},re5_r_{op,oop},re5_t_{op,oop},re5_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r5),n_ec2_{op,oop}(:,:,r5),n_ec3_{op,oop}(:,:,r5));
        end
    end
    [dr5,ddr5]=sel2_av2(re5_l_,re5_r_,84,chs,jji,l,r);
    [fl5,fr5,ft5,do5]=fea_sel3(re5_l_,re5_r_,re5_t_,re5_d_,ddr5,jji,chs);
    mi5(chs,:)=dr5;
    mii5(chs,:)=ddr5;
    lcsp5(:,:,chs)=[fl5];
    rcsp5(:,:,chs)=[fr5];
    tcsp5(:,:,chs)=[ft5];
    dif_t5(chs,:)=[do5];
    %% re6
    r6=[1,4,6,7,8];
    for op=chs:chs
        for oop=1:jji
            [re6_l_{op,oop},re6_r_{op,oop},re6_t_{op,oop},re6_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r6),n_ec2_{op,oop}(:,:,r6),n_ec3_{op,oop}(:,:,r6));
        end
    end
    [dr6,ddr6]=sel2_av2(re6_l_,re6_r_,84,chs,jji,l,r);
    [fl6,fr6,ft6,do6]=fea_sel3(re6_l_,re6_r_,re6_t_,re6_d_,ddr6,jji,chs);
    mi6(chs,:)=dr6;
    mii6(chs,:)=ddr6;
    lcsp6(:,:,chs)=[fl6];
    rcsp6(:,:,chs)=[fr6];
    tcsp6(:,:,chs)=[ft6];
    dif_t6(chs,:)=[do6];
    %% re7
    r7=[4,5,6,7,8,9,10];
    for op=chs:chs
        for oop=1:jji
            [re7_l_{op,oop},re7_r_{op,oop},re7_t_{op,oop},re7_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r7),n_ec2_{op,oop}(:,:,r7),n_ec3_{op,oop}(:,:,r7));
        end
    end
    [dr7,ddr7]=sel2_av2(re7_l_,re7_r_,84,chs,jji,l,r);
    [fl7,fr7,ft7,do7]=fea_sel3(re7_l_,re7_r_,re7_t_,re7_d_,ddr7,jji,chs);
    mi7(chs,:)=dr7;
    mii7(chs,:)=ddr7;
    lcsp7(:,:,chs)=[fl7];
    rcsp7(:,:,chs)=[fr7];
    tcsp7(:,:,chs)=[ft7];
    dif_t7(chs,:)=[do7];
    %% re8
    r8=[6,7,8,10];
    for op=chs:chs
        for oop=1:jji
            [re8_l_{op,oop},re8_r_{op,oop},re8_t_{op,oop},re8_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r8),n_ec2_{op,oop}(:,:,r8),n_ec3_{op,oop}(:,:,r8));
        end
    end
    [dr8,ddr8]=sel2_av2(re8_l_,re8_r_,84,chs,jji,l,r);
    [fl8,fr8,ft8,do8]=fea_sel3(re8_l_,re8_r_,re8_t_,re8_d_,ddr8,jji,chs);
    mi8(chs,:)=dr8;
    mii8(chs,:)=ddr8;
    lcsp8(:,:,chs)=[fl8];
    rcsp8(:,:,chs)=[fr8];
    tcsp8(:,:,chs)=[ft8];
    dif_t8(chs,:)=[do8];
    %% re9
    r9=[5,9,10,11];
    for op=chs:chs
        for oop=1:jji
            [re9_l_{op,oop},re9_r_{op,oop},re9_t_{op,oop},re9_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r9),n_ec2_{op,oop}(:,:,r9),n_ec3_{op,oop}(:,:,r9));
        end
    end
    [dr9,ddr9]=sel2_av2(re9_l_,re9_r_,84,chs,jji,l,r);
    [fl9,fr9,ft9,do9]=fea_sel3(re9_l_,re9_r_,re9_t_,re9_d_,ddr9,jji,chs);
    mi9(chs,:)=dr9;
    mii9(chs,:)=ddr9;
    lcsp9(:,:,chs)=[fl9];
    rcsp9(:,:,chs)=[fr9];
    tcsp9(:,:,chs)=[ft9];
    dif_t9(chs,:)=[do9];
    %% re10
    r10=[7,8,9,10,16,18];
    for op=chs:chs
        for oop=1:jji
            [re10_l_{op,oop},re10_r_{op,oop},re10_t_{op,oop},re10_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r10),n_ec2_{op,oop}(:,:,r10),n_ec3_{op,oop}(:,:,r10));
        end
    end
    [dr10,ddr10]=sel2_av2(re10_l_,re10_r_,84,chs,jji,l,r);
    [fl10,fr10,ft10,do10]=fea_sel3(re10_l_,re10_r_,re10_t_,re10_d_,ddr10,jji,chs);
    mi10(chs,:)=dr10;
    mii10(chs,:)=ddr10;
    lcsp10(:,:,chs)=[fl10];
    rcsp10(:,:,chs)=[fr10];
    tcsp10(:,:,chs)=[ft10];
    dif_t10(chs,:)=[do10];
    %% re11
    r11=[9,11,12,14,16];
    for op=chs:chs
        for oop=1:jji
            [re11_l_{op,oop},re11_r_{op,oop},re11_t_{op,oop},re11_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r11),n_ec2_{op,oop}(:,:,r11),n_ec3_{op,oop}(:,:,r11));
        end
    end
    [dr11,ddr11]=sel2_av2(re11_l_,re11_r_,84,chs,jji,l,r);
    [fl11,fr11,ft11,do11]=fea_sel3(re11_l_,re11_r_,re11_t_,re11_d_,ddr11,jji,chs);
    mi11(chs,:)=dr11;
    mii11(chs,:)=ddr11;
    lcsp11(:,:,chs)=[fl11];
    rcsp11(:,:,chs)=[fr11];
    tcsp11(:,:,chs)=[ft11];
    dif_t11(chs,:)=[do11];
    %% re12
    r12=[9,11,12,13,14];
    for op=chs:chs
        for oop=1:jji
            [re12_l_{op,oop},re12_r_{op,oop},re12_t_{op,oop},re12_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r12),n_ec2_{op,oop}(:,:,r12),n_ec3_{op,oop}(:,:,r12));
        end
    end
    [dr12,ddr12]=sel2_av2(re12_l_,re12_r_,84,chs,jji,l,r);
    [fl12,fr12,ft12,do12]=fea_sel3(re12_l_,re12_r_,re12_t_,re12_d_,ddr12,jji,chs);
    mi12(chs,:)=dr12;
    mii12(chs,:)=ddr12;
    lcsp12(:,:,chs)=[fl12];
    rcsp12(:,:,chs)=[fr12];
    tcsp12(:,:,chs)=[ft12];
    dif_t12(chs,:)=[do12];
    %% re13
    r13=[12,13,14,15];
    for op=chs:chs
        for oop=1:jji
            [re13_l_{op,oop},re13_r_{op,oop},re13_t_{op,oop},re13_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r13),n_ec2_{op,oop}(:,:,r13),n_ec3_{op,oop}(:,:,r13));
        end
    end
    [dr13,ddr13]=sel2_av2(re13_l_,re13_r_,84,chs,jji,l,r);
    [fl13,fr13,ft13,do13]=fea_sel3(re13_l_,re13_r_,re13_t_,re13_d_,ddr13,jji,chs);
    mi13(chs,:)=dr13;
    mii13(chs,:)=ddr13;
    lcsp13(:,:,chs)=[fl13];
    rcsp13(:,:,chs)=[fr13];
    tcsp13(:,:,chs)=[ft13];
    dif_t13(chs,:)=[do13];
    %% re14
    r14=[11,12,13,14,15,16,17];
    for op=chs:chs
        for oop=1:jji
            [re14_l_{op,oop},re14_r_{op,oop},re14_t_{op,oop},re14_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r14),n_ec2_{op,oop}(:,:,r14),n_ec3_{op,oop}(:,:,r14));
        end
    end
    [dr14,ddr14]=sel2_av2(re14_l_,re14_r_,84,chs,jji,l,r);
    [fl14,fr14,ft14,do14]=fea_sel3(re14_l_,re14_r_,re14_t_,re14_d_,ddr14,jji,chs);
    mi14(chs,:)=dr14;
    mii14(chs,:)=ddr14;
    lcsp14(:,:,chs)=[fl14];
    rcsp14(:,:,chs)=[fr14];
    tcsp14(:,:,chs)=[ft14];
    dif_t14(chs,:)=[do14];
    %% re15
    r15=[13,14,15,17];
    for op=chs:chs
        for oop=1:jji
            [re15_l_{op,oop},re15_r_{op,oop},re15_t_{op,oop},re15_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r15),n_ec2_{op,oop}(:,:,r15),n_ec3_{op,oop}(:,:,r15));
        end
    end
    [dr15,ddr15]=sel2_av2(re15_l_,re15_r_,84,chs,jji,l,r);
    [fl15,fr15,ft15,do15]=fea_sel3(re15_l_,re15_r_,re15_t_,re15_d_,ddr15,jji,chs);
    mi15(chs,:)=dr15;
    mii15(chs,:)=ddr15;
    lcsp15(:,:,chs)=[fl15];
    rcsp15(:,:,chs)=[fr15];
    tcsp15(:,:,chs)=[ft15];
    dif_t15(chs,:)=[do15];
    %% re16
    r16=[9,10,11,14,16,17,18];
    for op=chs:chs
        for oop=1:jji
            [re16_l_{op,oop},re16_r_{op,oop},re16_t_{op,oop},re16_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r16),n_ec2_{op,oop}(:,:,r16),n_ec3_{op,oop}(:,:,r16));
        end
    end
    [dr16,ddr16]=sel2_av2(re16_l_,re16_r_,84,chs,jji,l,r);
    [fl16,fr16,ft16,do16]=fea_sel3(re16_l_,re16_r_,re16_t_,re16_d_,ddr16,jji,chs);
    mi16(chs,:)=dr16;
    mii16(chs,:)=ddr16;
    lcsp16(:,:,chs)=[fl16];
    rcsp16(:,:,chs)=[fr16];
    tcsp16(:,:,chs)=[ft16];
    dif_t16(chs,:)=[do16];
    %% re17
    r17=[14,15,16,17,18];
    for op=chs:chs
        for oop=1:jji
            [re17_l_{op,oop},re17_r_{op,oop},re17_t_{op,oop},re17_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r17),n_ec2_{op,oop}(:,:,r17),n_ec3_{op,oop}(:,:,r17));
        end
    end
    [dr17,ddr17]=sel2_av2(re17_l_,re17_r_,84,chs,jji,l,r);
    [fl17,fr17,ft17,do17]=fea_sel3(re17_l_,re17_r_,re17_t_,re17_d_,ddr17,jji,chs);
    mi17(chs,:)=dr17;
    mii17(chs,:)=ddr17;
    lcsp17(:,:,chs)=[fl17];
    rcsp17(:,:,chs)=[fr17];
    tcsp17(:,:,chs)=[ft17];
    dif_t17(chs,:)=[do17];
    %% re18
    r18=[10,16,17,18];
    for op=chs:chs
        for oop=1:jji
            [re18_l_{op,oop},re18_r_{op,oop},re18_t_{op,oop},re18_d_{op,oop}]=pyk(l,r,tr,n_ec1_{op,oop}(:,:,r18),n_ec2_{op,oop}(:,:,r18),n_ec3_{op,oop}(:,:,r18));
        end
    end
    [dr18,ddr18]=sel2_av2(re18_l_,re18_r_,84,chs,jji,l,r);
    [fl18,fr18,ft18,do18]=fea_sel3(re18_l_,re18_r_,re18_t_,re18_d_,ddr18,jji,chs);
    mi18(chs,:)=dr18;
    mii18(chs,:)=ddr18;
    lcsp18(:,:,chs)=[fl18];
    rcsp18(:,:,chs)=[fr18];
    tcsp18(:,:,chs)=[ft18];
    dif_t18(chs,:)=[do18];
end
%%
[ffl1,ffr1,fft1,ddo1]=ega(mi1,lcsp1,rcsp1,tcsp1,dif_t1); % filter banks selection based on MIBIF
[ffl2,ffr2,fft2,ddo2]=ega(mi2,lcsp2,rcsp2,tcsp2,dif_t2);
[ffl3,ffr3,fft3,ddo3]=ega(mi3,lcsp3,rcsp3,tcsp3,dif_t3);
[ffl4,ffr4,fft4,ddo4]=ega(mi4,lcsp4,rcsp4,tcsp4,dif_t4);
[ffl5,ffr5,fft5,ddo5]=ega(mi5,lcsp5,rcsp5,tcsp5,dif_t5);
[ffl6,ffr6,fft6,ddo6]=ega(mi6,lcsp6,rcsp6,tcsp6,dif_t6);
[ffl7,ffr7,fft7,ddo7]=ega(mi7,lcsp7,rcsp7,tcsp7,dif_t7);
[ffl8,ffr8,fft8,ddo8]=ega(mi8,lcsp8,rcsp8,tcsp8,dif_t8);
[ffl9,ffr9,fft9,ddo9]=ega(mi9,lcsp9,rcsp9,tcsp9,dif_t9);
[ffl10,ffr10,fft10,ddo10]=ega(mi10,lcsp10,rcsp10,tcsp10,dif_t10);
[ffl11,ffr11,fft11,ddo11]=ega(mi11,lcsp11,rcsp11,tcsp11,dif_t11);
[ffl12,ffr12,fft12,ddo12]=ega(mi12,lcsp12,rcsp12,tcsp12,dif_t12);
[ffl13,ffr13,fft13,ddo13]=ega(mi13,lcsp13,rcsp13,tcsp13,dif_t13);
[ffl14,ffr14,fft14,ddo14]=ega(mi14,lcsp14,rcsp14,tcsp14,dif_t14);
[ffl15,ffr15,fft15,ddo15]=ega(mi15,lcsp15,rcsp15,tcsp15,dif_t15);
[ffl16,ffr16,fft16,ddo16]=ega(mi16,lcsp16,rcsp16,tcsp16,dif_t16);
[ffl17,ffr17,fft17,ddo17]=ega(mi17,lcsp17,rcsp17,tcsp17,dif_t17);
[ffl18,ffr18,fft18,ddo18]=ega(mi18,lcsp18,rcsp18,tcsp18,dif_t18);

eig_d=[ddo1,ddo2,ddo3,ddo4,ddo5,ddo6,ddo7,ddo8,ddo9,ddo10,ddo11,ddo12,ddo13,ddo14,ddo15,ddo16,ddo17,ddo18]; %Eigenvalue score of each local region
point_set=length(find(eig_d>=max(eig_d)-min(eig_d))) % eigenvalue threshold
[to1,to2]=sort(eig_d);
llcsp=[ffl1,ffl2,ffl3,ffl4,ffl5,ffl6,ffl7,ffl8,ffl9,ffl10,ffl11,ffl12,ffl13,ffl14,ffl15,ffl16,ffl17,ffl18];
rrcsp=[ffr1,ffr2,ffr3,ffr4,ffr5,ffr6,ffr7,ffr8,ffr9,ffr10,ffr11,ffr12,ffr13,ffr14,ffr15,ffr16,ffr17,ffr18];
ttcsp=[fft1,fft2,fft3,fft4,fft5,fft6,fft7,fft8,fft9,fft10,fft11,fft12,fft13,fft14,fft15,fft16,fft17,fft18];

[fi1,fi2]=sort(eig_d,'descend');
fffl{1}=ffl1;fffl{2}=ffl2;fffl{3}=ffl3;fffl{4}=ffl4;fffl{5}=ffl5;fffl{6}=ffl6;fffl{7}=ffl7;fffl{8}=ffl8;fffl{9}=ffl9;
fffl{10}=ffl10;fffl{11}=ffl11;fffl{12}=ffl12;fffl{13}=ffl13;fffl{14}=ffl14;fffl{15}=ffl15;fffl{16}=ffl16;fffl{17}=ffl17;
fffl{18}=ffl18;
fffr{1}=ffr1;fffr{2}=ffr2;fffr{3}=ffr3;fffr{4}=ffr4;fffr{5}=ffr5;fffr{6}=ffr6;fffr{7}=ffr7;fffr{8}=ffr8;fffr{9}=ffr9;
fffr{10}=ffr10;fffr{11}=ffr11;fffr{12}=ffr12;fffr{13}=ffr13;fffr{14}=ffr14;fffr{15}=ffr15;fffr{16}=ffr16;fffr{17}=ffr17;
fffr{18}=ffr18;
ffft{1}=fft1;ffft{2}=fft2;ffft{3}=fft3;ffft{4}=fft4;ffft{5}=fft5;ffft{6}=fft6;ffft{7}=fft7;ffft{8}=fft8;ffft{9}=fft9;
ffft{10}=fft10;ffft{11}=fft11;ffft{12}=fft12;ffft{13}=fft13;ffft{14}=fft14;ffft{15}=fft15;ffft{16}=fft16;ffft{17}=fft17;
ffft{18}=fft18;

for k=1:18
    f_fl(:,(4*k-3):4*k)=fffl{fi2(k)};
    f_fr(:,(4*k-3):4*k)=fffr{fi2(k)};
    f_ft(:,(4*k-3):4*k)=ffft{fi2(k)};
end
m_f_fl=mean(f_fl);
dfl=(f_fl-m_f_fl).^2;
m_f_fr=mean(f_fr);
dfr=(f_fr-m_f_fr).^2;
qq=0;
for k=4:4:72
    qq=qq+1;
    sww1(qq)=((sum(sum(dfl(:,1:k)))/42)/k);
    sww2(qq)=((sum(sum(dfr(:,1:k)))/42)/k);
    sww(qq)=(sww1(qq)+sww2(qq))/2;
    sdd(qq)=sum((m_f_fl(1,1:k)-m_f_fr(1,1:k)).^2)/k;
end
pp=0;
for p=point_set:point_set % feature selection using fisher ratio of FBCSP feature
    clcsp=[];crcsp=[];ctcsp=[];go1=[];sav=[];
    pp=pp+1;
    trs=sdd(p)/sww(p);
    qq=0;
    for k=1:4:69
        qq=qq+1;
        sw1(qq)=((sum(sum(dfl(:,k:k+3)))/42)/4);
        sw2(qq)=((sum(sum(dfr(:,k:k+3)))/42)/4);
        sw(qq)=(sw1(qq)+sw2(qq))/2;
        sd(qq)=sum((m_f_fl(1,k:k+3)-m_f_fr(1,k:k+3)).^2)/4;
    end
    go1=(sd./sw)-trs;
    ii=0;
    for k=1:18
        if go1(k)>0
            ii=ii+1;
            sav(ii)=k;
        end
    end
    for kkk=1:length(sav)
        clcsp(:,(4*kkk-3):4*kkk)=fffl{fi2(sav(kkk))};
        crcsp(:,(4*kkk-3):4*kkk)=fffr{fi2(sav(kkk))};
        ctcsp(:,(4*kkk-3):4*kkk)=ffft{fi2(sav(kkk))};
    end
    vf=[clcsp;crcsp];
    lvf=[ones(l,1);ones(r,1)+1];
    options.MaxIter = 100000;
    SVMStruct = svmtrain(vf,lvf,'Options', options);
    mdl=fitcsvm(vf,lvf);%,'BoxConstraint',0.26097,'KernelScale',0.021833);
    tvl=true_y(85:end);
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
    
    aaccy1(pp)=correct1/length(result1)*100;
    aaccy(pp)=correct/length(result)*100;
end
