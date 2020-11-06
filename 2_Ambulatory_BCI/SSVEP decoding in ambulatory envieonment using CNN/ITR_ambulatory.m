%% Get ITR
N = 3; % the number of class
P = excel_ACC;
duration = 1;
C = 60./duration';

ITR = (log2(N)+P.*log2(P)+(1-P).*log2((1-P)./(N-1))).*C;
% disp(ITR)
disp('Average of ITR')
fprintf('%.2f\n',mean(ITR))

