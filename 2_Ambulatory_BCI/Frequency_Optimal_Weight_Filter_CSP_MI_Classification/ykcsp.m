function [hj,diff] =ykcsp(co11,co22)
rm=mean(co11,3);
rma=mean(co22,3);
R=(rm+rma);
[U,Lambda] = eig(R);
% [Lambda,ind] = sort(diag(Lambda),'descend');
% U=U(:,ind);
P=sqrt(inv((Lambda)))*U';
S{1}=P*rm*P';
S{2}=P*rma*P';

q=P*rm*P';
qq=P*rma*P';
check=(q+qq);

[C,CC]=eig(q);
mk=max(diag(CC));
mmk=min(diag(CC));

[Ca,CCa]=eig(qq);
nw=P'*C;
BB=nw'*rm*nw;
BBB=nw'*rma*nw;
[amp1,loc1]=max(diag(BB));
[amp2,loc2]=max(diag(BBB));
BB(loc1,loc1)=0;
BBB(loc2,loc2)=0;
[amp3,loc3]=max(diag(BB));
[amp4,loc4]=max(diag(BBB));
wn=nw';
hj(:,:)=[wn(loc1,:);wn(loc2,:)];
diff= mk-mmk;