clear all
load('data_set_IVa_av');
load('true_labels_av');
cnt= 0.1*double(cnt);
cue=mrk.pos;
yy=mrk.y;
cue=transpose(cue);
temp=[];
cnt=cnt;
data_training=84;
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
[bbb,aaa]=butter(4,[9/50 30/50]);
vf=[];n_ec1=[];n_ec2=[];n_ec3=[];ect1=[];ect2=[];ect3=[];result=[];fe1=[];fe2=[];fe3=[];result1=[];
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
%% 1
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,1);n_ec1(trial,:,2);n_ec1(trial,:,4);n_ec1(trial,:,6)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,1);n_ec2(trial,:,2);n_ec2(trial,:,4);n_ec2(trial,:,6)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,1);n_ec3(trial,:,2);n_ec3(trial,:,4);n_ec3(trial,:,6)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff1]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,1);n_ec1(trial,:,2);n_ec1(trial,:,4);n_ec1(trial,:,6)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_1=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,1);n_ec2(trial,:,2);n_ec2(trial,:,4);n_ec2(trial,:,6)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_1=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,1);n_ec3(trial,:,2);n_ec3(trial,:,4);n_ec3(trial,:,6)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_1=[max_f_t;min_f_t]';
end
%% 2
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,2);n_ec1(trial,:,1);n_ec1(trial,:,4);n_ec1(trial,:,3)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,2);n_ec2(trial,:,1);n_ec2(trial,:,4);n_ec2(trial,:,3)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,2);n_ec3(trial,:,1);n_ec3(trial,:,4);n_ec3(trial,:,3)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff2]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,2);n_ec1(trial,:,1);n_ec1(trial,:,4);n_ec1(trial,:,3)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_2=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,2);n_ec2(trial,:,1);n_ec2(trial,:,4);n_ec2(trial,:,3)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_2=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,2);n_ec3(trial,:,1);n_ec3(trial,:,4);n_ec3(trial,:,3)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_2=[max_f_t;min_f_t]';
end
%% 3
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,2);n_ec1(trial,:,5);n_ec1(trial,:,4);n_ec1(trial,:,9);n_ec1(trial,:,3)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,2);n_ec2(trial,:,5);n_ec2(trial,:,4);n_ec2(trial,:,9);n_ec2(trial,:,3)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,2);n_ec3(trial,:,5);n_ec3(trial,:,4);n_ec3(trial,:,9);n_ec3(trial,:,3)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff3]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,2);n_ec1(trial,:,5);n_ec1(trial,:,4);n_ec1(trial,:,9);n_ec1(trial,:,3)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_3=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,2);n_ec2(trial,:,5);n_ec2(trial,:,4);n_ec2(trial,:,9);n_ec2(trial,:,3)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_3=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,2);n_ec3(trial,:,5);n_ec3(trial,:,4);n_ec3(trial,:,9);n_ec3(trial,:,3)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_3=[max_f_t;min_f_t]';
end
%% 4
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,1);n_ec1(trial,:,2);n_ec1(trial,:,3);n_ec1(trial,:,5);n_ec1(trial,:,6);n_ec1(trial,:,7);n_ec1(trial,:,4)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,1);n_ec2(trial,:,2);n_ec2(trial,:,3);n_ec2(trial,:,5);n_ec2(trial,:,6);n_ec2(trial,:,7);n_ec2(trial,:,4)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,1);n_ec3(trial,:,2);n_ec3(trial,:,3);n_ec3(trial,:,5);n_ec3(trial,:,6);n_ec3(trial,:,7);n_ec3(trial,:,4)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff4]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,1);n_ec1(trial,:,2);n_ec1(trial,:,3);n_ec1(trial,:,5);n_ec1(trial,:,6);n_ec1(trial,:,7);n_ec1(trial,:,4)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_4=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,1);n_ec2(trial,:,2);n_ec2(trial,:,3);n_ec2(trial,:,5);n_ec2(trial,:,6);n_ec2(trial,:,7);n_ec2(trial,:,4)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_4=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,1);n_ec3(trial,:,2);n_ec3(trial,:,3);n_ec3(trial,:,5);n_ec3(trial,:,6);n_ec3(trial,:,7);n_ec3(trial,:,4)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_4=[max_f_t;min_f_t]';
end
%% 5
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,5);n_ec1(trial,:,3);n_ec1(trial,:,4);n_ec1(trial,:,7);n_ec1(trial,:,9)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,5);n_ec2(trial,:,3);n_ec2(trial,:,4);n_ec2(trial,:,7);n_ec2(trial,:,9)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,5);n_ec3(trial,:,3);n_ec3(trial,:,4);n_ec3(trial,:,7);n_ec3(trial,:,9)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff5]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,5);n_ec1(trial,:,3);n_ec1(trial,:,4);n_ec1(trial,:,7);n_ec1(trial,:,9)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_5=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,5);n_ec2(trial,:,3);n_ec2(trial,:,4);n_ec2(trial,:,7);n_ec2(trial,:,9)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_5=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,5);n_ec3(trial,:,3);n_ec3(trial,:,4);n_ec3(trial,:,7);n_ec3(trial,:,9)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_5=[max_f_t;min_f_t]';
end
%% 6
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,6);n_ec1(trial,:,1);n_ec1(trial,:,4);n_ec1(trial,:,7);n_ec1(trial,:,8)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,6);n_ec2(trial,:,1);n_ec2(trial,:,4);n_ec2(trial,:,7);n_ec2(trial,:,8)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,6);n_ec3(trial,:,1);n_ec3(trial,:,4);n_ec3(trial,:,7);n_ec3(trial,:,8)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff6]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,6);n_ec1(trial,:,1);n_ec1(trial,:,4);n_ec1(trial,:,7);n_ec1(trial,:,8)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_6=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,6);n_ec2(trial,:,1);n_ec2(trial,:,4);n_ec2(trial,:,7);n_ec2(trial,:,8)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_6=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,6);n_ec3(trial,:,1);n_ec3(trial,:,4);n_ec3(trial,:,7);n_ec3(trial,:,8)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_6=[max_f_t;min_f_t]';
end
%% 7
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,7);n_ec1(trial,:,4);n_ec1(trial,:,5);n_ec1(trial,:,6);n_ec1(trial,:,8);n_ec1(trial,:,9);n_ec1(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,7);n_ec2(trial,:,4);n_ec2(trial,:,5);n_ec2(trial,:,6);n_ec2(trial,:,8);n_ec2(trial,:,9);n_ec2(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,7);n_ec3(trial,:,4);n_ec3(trial,:,5);n_ec3(trial,:,6);n_ec3(trial,:,8);n_ec3(trial,:,9);n_ec3(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff7]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,7);n_ec1(trial,:,4);n_ec1(trial,:,5);n_ec1(trial,:,6);n_ec1(trial,:,8);n_ec1(trial,:,9);n_ec1(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_7=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,7);n_ec2(trial,:,4);n_ec2(trial,:,5);n_ec2(trial,:,6);n_ec2(trial,:,8);n_ec2(trial,:,9);n_ec2(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_7=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,7);n_ec3(trial,:,4);n_ec3(trial,:,5);n_ec3(trial,:,6);n_ec3(trial,:,8);n_ec3(trial,:,9);n_ec3(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_7=[max_f_t;min_f_t]';
end
%% 8
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,8);n_ec1(trial,:,6);n_ec1(trial,:,7);n_ec1(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,8);n_ec2(trial,:,6);n_ec2(trial,:,7);n_ec2(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,8);n_ec3(trial,:,6);n_ec3(trial,:,7);n_ec3(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff8]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,8);n_ec1(trial,:,6);n_ec1(trial,:,7);n_ec1(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_8=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,8);n_ec2(trial,:,6);n_ec2(trial,:,7);n_ec2(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_8=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,8);n_ec3(trial,:,6);n_ec3(trial,:,7);n_ec3(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_8=[max_f_t;min_f_t]';
end
%% 9
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,9);n_ec1(trial,:,5);n_ec1(trial,:,11);n_ec1(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,9);n_ec2(trial,:,5);n_ec2(trial,:,11);n_ec2(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,9);n_ec3(trial,:,5);n_ec3(trial,:,11);n_ec3(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff9]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,9);n_ec1(trial,:,5);n_ec1(trial,:,11);n_ec1(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_9=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,9);n_ec2(trial,:,5);n_ec2(trial,:,11);n_ec2(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_9=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,9);n_ec3(trial,:,5);n_ec3(trial,:,11);n_ec3(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_9=[max_f_t;min_f_t]';
end
%% 10
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,10);n_ec1(trial,:,9);n_ec1(trial,:,7);n_ec1(trial,:,16);n_ec1(trial,:,8);n_ec1(trial,:,18)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,10);n_ec2(trial,:,9);n_ec2(trial,:,7);n_ec2(trial,:,16);n_ec2(trial,:,8);n_ec2(trial,:,18)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,10);n_ec3(trial,:,9);n_ec3(trial,:,7);n_ec3(trial,:,16);n_ec3(trial,:,8);n_ec3(trial,:,18)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff10]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,10);n_ec1(trial,:,9);n_ec1(trial,:,7);n_ec1(trial,:,16);n_ec1(trial,:,8);n_ec1(trial,:,18)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_10=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,10);n_ec2(trial,:,9);n_ec2(trial,:,7);n_ec2(trial,:,16);n_ec2(trial,:,8);n_ec2(trial,:,18)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_10=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,10);n_ec3(trial,:,9);n_ec3(trial,:,7);n_ec3(trial,:,16);n_ec3(trial,:,8);n_ec3(trial,:,18)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_10=[max_f_t;min_f_t]';
end
%% 11
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,12);n_ec1(trial,:,14);n_ec1(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,12);n_ec2(trial,:,14);n_ec2(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,12);n_ec3(trial,:,14);n_ec3(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff11]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,12);n_ec1(trial,:,14);n_ec1(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_11=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,12);n_ec2(trial,:,14);n_ec2(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_11=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,12);n_ec3(trial,:,14);n_ec3(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_11=[max_f_t;min_f_t]';
end
%% 12
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,12)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,12)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,12)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff12]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,12)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_12=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,12)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_12=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,12)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_12=[max_f_t;min_f_t]';
end
%% 13
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,15);n_ec1(trial,:,12);n_ec1(trial,:,13);n_ec1(trial,:,14)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,15);n_ec2(trial,:,12);n_ec2(trial,:,13);n_ec2(trial,:,14)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,15);n_ec3(trial,:,12);n_ec3(trial,:,13);n_ec3(trial,:,14)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff13]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,15);n_ec1(trial,:,12);n_ec1(trial,:,13);n_ec1(trial,:,14)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_13=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,15);n_ec2(trial,:,12);n_ec2(trial,:,13);n_ec2(trial,:,14)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_13=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,15);n_ec3(trial,:,12);n_ec3(trial,:,13);n_ec3(trial,:,14)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_13=[max_f_t;min_f_t]';
end
%% 14
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,11);n_ec1(trial,:,12);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,15);n_ec1(trial,:,16);n_ec1(trial,:,17)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,11);n_ec2(trial,:,12);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,15);n_ec2(trial,:,16);n_ec2(trial,:,17)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,11);n_ec3(trial,:,12);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,15);n_ec3(trial,:,16);n_ec3(trial,:,17)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff14]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,11);n_ec1(trial,:,12);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,15);n_ec1(trial,:,16);n_ec1(trial,:,17)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_14=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,11);n_ec2(trial,:,12);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,15);n_ec2(trial,:,16);n_ec2(trial,:,17)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_14=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,11);n_ec3(trial,:,12);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,15);n_ec3(trial,:,16);n_ec3(trial,:,17)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_14=[max_f_t;min_f_t]';
end
%% 15
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,15);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,17)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,15);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,17)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,15);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,17)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff15]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,15);n_ec1(trial,:,13);n_ec1(trial,:,14);n_ec1(trial,:,17)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_15=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,15);n_ec2(trial,:,13);n_ec2(trial,:,14);n_ec2(trial,:,17)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_15=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,15);n_ec3(trial,:,13);n_ec3(trial,:,14);n_ec3(trial,:,17)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_15=[max_f_t;min_f_t]';
end
%% 16
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,16);n_ec1(trial,:,14);n_ec1(trial,:,17);n_ec1(trial,:,18);n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,10)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,16);n_ec2(trial,:,14);n_ec2(trial,:,17);n_ec2(trial,:,18);n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,10)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,16);n_ec3(trial,:,14);n_ec3(trial,:,17);n_ec3(trial,:,18);n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,10)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff16]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,16);n_ec1(trial,:,14);n_ec1(trial,:,17);n_ec1(trial,:,18);n_ec1(trial,:,11);n_ec1(trial,:,9);n_ec1(trial,:,10)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_16=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,16);n_ec2(trial,:,14);n_ec2(trial,:,17);n_ec2(trial,:,18);n_ec2(trial,:,11);n_ec2(trial,:,9);n_ec2(trial,:,10)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_16=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,16);n_ec3(trial,:,14);n_ec3(trial,:,17);n_ec3(trial,:,18);n_ec3(trial,:,11);n_ec3(trial,:,9);n_ec3(trial,:,10)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_16=[max_f_t;min_f_t]';
end
%% 17
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,14);n_ec1(trial,:,15);n_ec1(trial,:,17);n_ec1(trial,:,18);n_ec1(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,14);n_ec2(trial,:,15);n_ec2(trial,:,17);n_ec2(trial,:,18);n_ec2(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,14);n_ec3(trial,:,15);n_ec3(trial,:,17);n_ec3(trial,:,18);n_ec3(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff17]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,14);n_ec1(trial,:,15);n_ec1(trial,:,17);n_ec1(trial,:,18);n_ec1(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_17=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,14);n_ec2(trial,:,15);n_ec2(trial,:,17);n_ec2(trial,:,18);n_ec2(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_17=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,14);n_ec3(trial,:,15);n_ec3(trial,:,17);n_ec3(trial,:,18);n_ec3(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_17=[max_f_t;min_f_t]';
end
%% 18
for ry=1:1
    leg1_ch=[];co1=[];leg2_ch=[];co2=[];leg3_ch=[];co3=[];max_f_l=[];max_f_r=[];max_f_t=[];min_f_l=[];min_f_r=[];min_f_t=[];
    for trial=1:l
        leg1_ch(:,:)=[n_ec1(trial,:,18);n_ec1(trial,:,10);n_ec1(trial,:,17);n_ec1(trial,:,16)];
        co1(:,:,trial)=leg1_ch*leg1_ch'/trace(leg1_ch*leg1_ch');
    end
    for trial=1:r
        leg2_ch(:,:)=[n_ec2(trial,:,18);n_ec2(trial,:,10);n_ec2(trial,:,17);n_ec2(trial,:,16)];
        co2(:,:,trial)=leg2_ch*leg2_ch'/trace(leg2_ch*leg2_ch');
    end
    for trial=1:tr
        leg3_ch(:,:)=[n_ec3(trial,:,18);n_ec3(trial,:,10);n_ec3(trial,:,17);n_ec3(trial,:,16)];
        co3(:,:,trial)=leg3_ch*leg3_ch'/trace(leg3_ch*leg3_ch');
    end
    [rt,diff18]=ykcsp(co1,co2);
    for trial=1:l
        fe1=rt*[n_ec1(trial,:,18);n_ec1(trial,:,10);n_ec1(trial,:,17);n_ec1(trial,:,16)];
        max_f_l(trial)=var(fe1(1,:))/(var(fe1(1,:))+var(fe1(2,:)));
        min_f_l(trial)=var(fe1(2,:))/(var(fe1(1,:))+var(fe1(2,:)));
    end
    f_l_18=[max_f_l;min_f_l]';
    for trial=1:r
        fe2=rt*[n_ec2(trial,:,18);n_ec2(trial,:,10);n_ec2(trial,:,17);n_ec2(trial,:,16)];
        max_f_r(trial)=var(fe2(1,:))/(var(fe2(1,:))+var(fe2(2,:)));
        min_f_r(trial)=var(fe2(2,:))/(var(fe2(1,:))+var(fe2(2,:)));
    end
    f_r_18=[max_f_r;min_f_r]';
    for trial=1:tr
        fe3=rt*[n_ec3(trial,:,18);n_ec3(trial,:,10);n_ec3(trial,:,17);n_ec3(trial,:,16)];
        max_f_t(trial)=var(fe3(1,:))/(var(fe3(1,:))+var(fe3(2,:)));
        min_f_t(trial)=var(fe3(2,:))/(var(fe3(1,:))+var(fe3(2,:)));
    end
    f_t_18=[max_f_t;min_f_t]';
end
%% sorting eigenvalue score
dif_t=[diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18];
[ko1,ko2]=sort(dif_t);
%% feature
lcsp=[f_l_1,f_l_2,f_l_3,f_l_4,f_l_5,f_l_6,f_l_7,f_l_8,f_l_9,f_l_10,f_l_11,f_l_12,f_l_13,f_l_14,f_l_15,f_l_16,f_l_17,f_l_18];
rcsp=[f_r_1,f_r_2,f_r_3,f_r_4,f_r_5,f_r_6,f_r_7,f_r_8,f_r_9,f_r_10,f_r_11,f_r_12,f_r_13,f_r_14,f_r_15,f_r_16,f_r_17,f_r_18];
tcsp=[f_t_1,f_t_2,f_t_3,f_t_4,f_t_5,f_t_6,f_t_7,f_t_8,f_t_9,f_t_10,f_t_11,f_t_12,f_t_13,f_t_14,f_t_15,f_t_16,f_t_17,f_t_18];
%% compute accy according to eigenvalue score
for yyu=1:length(ko2)
    clcsp=[];crcsp=[];ctcsp=[];
    for yu=yyu:length(ko2)
        teemp1=lcsp(:,2*ko2(yu)-1:2*ko2(yu));
        clcsp=[clcsp,teemp1];
        teemp2=rcsp(:,2*ko2(yu)-1:2*ko2(yu));
        crcsp=[crcsp,teemp2];
        teemp3=tcsp(:,2*ko2(yu)-1:2*ko2(yu));
        ctcsp=[ctcsp,teemp3];
    end
    vf=[clcsp;crcsp];
    lvf=[ones(l,1);ones(r,1)+1];
    options.MaxIter = 100000;
    SVMStruct = svmtrain(vf,lvf,'Options', options);
    tvl=true_y((data_training+1):end);
    result = svmclassify(SVMStruct,ctcsp);
    correct=0;
    for k=1:length(result)
        if result(k)==tvl(k)
            correct=correct+1;
        end
    end
    accy(yyu)=correct/length(result)*100;
end
