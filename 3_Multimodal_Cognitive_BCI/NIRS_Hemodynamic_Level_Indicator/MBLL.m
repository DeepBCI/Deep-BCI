function [con_hbo, con_hb]=MBLL(I1,I2,distance,w_long,w_short)
%%% 890, 750
% 
% % % 
% extin1_hb=1.2959;
% extin1_hbo=0.4646;
% extin2_hb=0.7861;
% extin2_hbo=1.1596;
% 
% L1=6*distance;
% L2=6*distance;
% 
% 
% A1=-log(I1/mean(I1));
% A2=-log(I2/mean(I2));
% 
% con_hbo=(extin1_hb*A2/L2-extin2_hb*A1/L1)/(extin1_hb*extin2_hbo-extin2_hb*extin1_hbo);
% con_hb=(extin1_hbo*A2/L2-extin2_hbo*A1/L1)/(extin1_hbo*extin2_hb-extin2_hbo*extin1_hb);
%%%%%%%%%%%%%%%%%%%%%
% 
I_w1=I1;
I_w2=I2;
ppf=60; dpf=6; L=distance;


dOD_w1 = -log(I_w1/mean(I_w1));
dOD_w2 = -log(I_w2/mean(I_w2));

E = GetExtinctions([w_long w_short]);  %E = [e at 830-HbO e at 830-HbR e at 830-Lipid
% e at 830-H2O e at 830-AA;
                                %     e at 690-HbO e at 690-HbR e at 690-Lipid
% e at 690-H2O e at 690-AA];

E=E(1:2,1:2);   %Only keep HbO/HbR parts

dOD_w1_L = dOD_w1 * ppf/(dpf*L);        %Pre-divide by pathlength (dpf,pvc
% etc)
dOD_w2_L = dOD_w2 * ppf/(dpf*L);

dOD_L = [dOD_w1_L dOD_w2_L];            %Concatinate the 2(or more)
% wavelengths

%I put pathlength into dOD_L so that I can preform one matrix inversion
% rather than                                          %one per #measurements.
% You could do inv(E*L) instead.

Einv = inv(E'*E)*E';            %Linear inversion operator (for 2 or more
% wavelengths)

HbO_HbR = Einv * dOD_L';
 %Solve for HbO and HbR (This is the least-squares solution for unlimited #
% of wavelengths)

con_hb = HbO_HbR(1,:);
con_hbo = HbO_HbR(2,:);


end