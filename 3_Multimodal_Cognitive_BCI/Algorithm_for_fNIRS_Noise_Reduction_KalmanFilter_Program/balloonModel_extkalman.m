function [F,JF]=balloonModel_extkalman(x,fin,C)
%constant=[E0,tau0,tauv,alpha,delt];
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
% 
E=1-(1-E0)^(1/fin);
% 
F=[v+delt.*(1/tauv)*(fin-v.^(1/alpha));...
    q+delt.*[(fin/tau0)*(E/E0-q/v)+(1/tauv)*(fin-v.^(1/alpha))*(q/v)];...
    p+delt.*[(1/tauv)*(fin-v.^(1/alpha))*(p/v)];
    a1;...
    a2];
JF=[1 - (delt*v^(1/alpha - 1))/(alpha*tauv),0,0,0,0;
    -delt*((q*(fin - v^(1/alpha)))/(tauv*v^2) - (fin*q)/(tau0*v^2) + (q*v^(1/alpha - 1))/(alpha*tauv*v)), 1 - delt*(fin/(tau0*v) - (fin - v^(1/alpha))/(tauv*v)),  0, 0, 0;
    - (delt*p*(fin - v^(1/alpha)))/(tauv*v^2) - (delt*p*v^(1/alpha - 1))/(alpha*tauv*v), 0, (delt*(fin - v^(1/alpha)))/(tauv*v) + 1, 0, 0;
    0,                                                      0,                                       0, 1, 0;
    0,                                                      0,                                       0, 0, 1];

F=double(F);
JF=double(JF);
% F_function(v,q,p,a1,a2)=[v+delt.*(1/tauv)*(fin-v.^(1/alpha));...
%     q+delt.*[(fin/tau0)*(E/E0-q/v)+(1/tauv)*(fin-v.^(1/alpha))*(q/v)];...
%     p+delt.*[(1/tauv)*(fin-v.^(1/alpha))*(p/v)];
%     a1;...
%     a2];

% 
% 
% JF_function(v,q,p,a1,a2)=jacobian(F_function,[v,q,p,a1,a2]);
        

        
% F=F_function(x(1),x(2),x(3),x(4),x(5));
% JF=JF_function(x(1),x(2),x(3),x(4),x(5));

end