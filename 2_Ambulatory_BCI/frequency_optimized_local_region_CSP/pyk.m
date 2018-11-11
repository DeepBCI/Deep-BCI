function [f_l_1,f_r_1,f_t_1,diff1_1] =pyk(l,r,tr,e1,e2,e3)
for trial=1:l
    leg1_ch_1(:,:)=e1(trial,:,:);
    o1(:,:,trial)=leg1_ch_1'*leg1_ch_1/trace(leg1_ch_1'*leg1_ch_1);
end
for trial=1:r
    leg2_ch_1(:,:)=e2(trial,:,:);
    o2(:,:,trial)=leg2_ch_1'*leg2_ch_1/trace(leg2_ch_1'*leg2_ch_1);
    
end
for trial=1:tr
    leg3_ch_1(:,:)=e3(trial,:,:);
    co3_1(:,:,trial)=leg3_ch_1'*leg3_ch_1/trace(leg3_ch_1'*leg3_ch_1);
    
end
[rt_1,diff1_1]=pykcsp(o1,o2);
for trial=1:l
    e11(:,:)=e1(trial,:,:);
        fe1_1=rt_1*e11';
        max_f_l_1(trial)=var(fe1_1(1,:))/(var(fe1_1(1,:))+var(fe1_1(2,:)));
        min_f_l_1(trial)=var(fe1_1(2,:))/(var(fe1_1(1,:))+var(fe1_1(2,:)));
end
for trial=1:r
    e22(:,:)=e2(trial,:,:);
        fe2_1=rt_1*e22';
        max_f_r_1(trial)=var(fe2_1(1,:))/(var(fe2_1(1,:))+var(fe2_1(2,:)));
        min_f_r_1(trial)=var(fe2_1(2,:))/(var(fe2_1(1,:))+var(fe2_1(2,:)));
end
for trial=1:tr
    e33(:,:)=e3(trial,:,:);
        fe3_1=rt_1*e33';
        max_f_t_1(trial)=var(fe3_1(1,:))/(var(fe3_1(1,:))+var(fe3_1(2,:)));
        min_f_t_1(trial)=var(fe3_1(2,:))/(var(fe3_1(1,:))+var(fe3_1(2,:)));
end
    f_l_1=[max_f_l_1;min_f_l_1]';
    f_r_1=[max_f_r_1;min_f_r_1]';
    f_t_1=[max_f_t_1;min_f_t_1]';