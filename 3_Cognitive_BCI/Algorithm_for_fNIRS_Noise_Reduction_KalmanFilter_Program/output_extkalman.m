function [H,JH]=output_extkalman(x,fin,C,ySS,hb) %hbo==1, hbr==2
%constant=[E0,V0,tau0,alpha,taus,tauf,eps];
%x:scalar
% C=[1,1,1,1,1,1,1];
% u=1;
% syms v q p a1 a2

v=x(1);
q=x(2);
p=x(3);
a1=x(4);
a2=x(5);
E0=C(1);
tau0=C(2);
tauv=C(3);
alpha=C(4);
delt=C(5);

E=1-(1-E0)^(1/fin);
if hb==1
H=a1*(p-q)+a2*ySS;
JH=[0,-a1,a1,p-q,ySS];
else
H=a1*q+a2*ySS;
JH=[0,a1,0,q,ySS];
end

% H_function(v,q,p,a1,a2)=a1*q+a2*ySS;

% 
% JH_function(v,q,p,a1,a2)=jacobian(H_function,[v,q,p,a1,a2]);
        
H=double(H);
JH=double(JH);

        
% H=H_function(x(1),x(2),x(3),x(4),x(5));
% JH=JH_function(x(1),x(2),x(3),x(4),x(5));

end