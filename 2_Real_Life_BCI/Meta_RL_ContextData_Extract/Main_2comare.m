%% For comparing results from simulations
clear;
whichSIMUL = 'ORI_ArbResult_w8228.mat';
load(whichSIMUL);

RREE=cell(1,22);
for i = 1 : 22
    for z = 1 : 100
        RREE{i} = [RREE{i} ; resulttt(z,i).NegLogLik_val];
        
        
    end
end
RERERE= cell(1,1);
for i = 1 : 22
RERERE{1} = [RERERE{1}  RREE{i}];
end

median(min(RERERE{1}))