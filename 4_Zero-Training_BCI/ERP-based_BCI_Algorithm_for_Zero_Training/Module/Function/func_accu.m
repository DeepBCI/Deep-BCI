function  Accu = func_accu( True_label, cf_out )
%EVAL_CALLOSS Summary of this function goes here
%   Detailed explanation goes here
%only if binary class
if size(True_label,1)==2 %logical
    tm=find(True_label(1,:)==1);
    tm2=find(True_label(2,:)==1);
    [nClass nTri]=size(True_label);
    bn_label=zeros(nTri,1);
    bn_label(tm)=-1;bn_label(tm2)=1;
    loss01=sign(cf_out)~=bn_label;
else
    Est_label= 1.5 + 0.5*sign(cf_out)';
    loss01=Est_label~=True_label;
end
loss=length(find(loss01)==1)/length(cf_out);
Accu=(1-loss)*100;
end

