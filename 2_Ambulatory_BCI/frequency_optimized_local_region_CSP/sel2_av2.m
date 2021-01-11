function [d,dd] =sel2_aw(f_l_ss,f_r_ss,tr,leg2,leg,l,r)
% dd=[f_l_s,f_r_s];
for mmk=leg2:leg2
for mk=1:leg
f_sum(:,2*mk-1:2*mk)=[f_l_ss{mmk,mk};f_r_ss{mmk,mk}];
f_l_s(:,2*mk-1:2*mk)=f_l_ss{mmk,mk};
f_r_s(:,2*mk-1:2*mk)=f_r_ss{mmk,mk};
end
for kk=1:2*leg
    s_l=[];
    s_r=[];
for k=1:tr
s_l(:,k)=f_sum(k,kk)-f_l_s(:,kk);
s_r(:,k)=f_sum(k,kk)-f_r_s(:,kk);
hopt_l(k)=(4/3/l)^(1/5)*std(s_l(:,k));
hopt_r(k)=(4/3/r)^(1/5)*std(s_r(:,k));
gk_l(:,k)=1/sqrt(2*pi)*exp(-((s_l(:,k)).^2./(2*hopt_l(k)^2)));
gk_r(:,k)=1/sqrt(2*pi)*exp(-((s_r(:,k)).^2./(2*hopt_r(k)^2)));
cp_l(k)=sum(gk_l(:,k))/l;
cp_r(k)=sum(gk_r(:,k))/r;
pf(k)=(cp_l(k)*(l/(l+r))+cp_r(k)*(r/(l+r)));
p_l_p(k)=cp_l(k)*(l/(l+r))/pf(k);
p_r_p(k)=cp_r(k)*(r/(l+r))/pf(k);

end
henp(kk)=-(sum(p_l_p.*log2(p_l_p))+sum(p_r_p.*log2(p_r_p)));
end
hw=-( ((l/(l+r))*log2((l/(l+r))))+((r/(l+r))*log2((r/(l+r)))) );
few=hw-henp;

[d,dd]=max(few);
end