function [fl,fr,ft,do] =fea_sel3(f_l_ss,f_r_ss,f_t_ss,f_d_ss,dr,leg,leg2)
mmk=leg2;
for mk=1:leg
f_sum(:,2*mk-1:2*mk)=[f_l_ss{mmk,mk};f_r_ss{mmk,mk}];
co1(:,2*mk-1:2*mk)=f_l_ss{mmk,mk};
co2(:,2*mk-1:2*mk)=f_r_ss{mmk,mk};
co3(:,2*mk-1:2*mk)=f_t_ss{mmk,mk};
co4(mk)=f_d_ss{mmk,mk};

end
fl=f_l_ss{mmk,ceil(dr/2)};
fr=f_r_ss{mmk,ceil(dr/2)};
ft=f_t_ss{mmk,ceil(dr/2)};
% if (mod(dr,2))~=0
%     fl=[co1(:,dr),co1(:,dr+1)];
% else
%     fl=[co1(:,dr-1),co1(:,dr)];
% end
% if (mod(dr,2))~=0
%     fr=[co2(:,dr),co2(:,dr+1)];
% else
%     fr=[co2(:,dr-1),co2(:,dr)];
% end
% if (mod(dr,2))~=0
%     ft=[co3(:,dr),co3(:,dr+1)];
% else
%     ft=[co3(:,dr-1),co3(:,dr)];
% end
do=co4(ceil(dr/2));
