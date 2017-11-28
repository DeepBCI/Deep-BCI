function [M,P] = EKF_predict(F,JF,P,Q)

    M=F;
    P=JF*P*JF'+Q;
    
end
    
    