function [M,P] = EKF_update(F,H,JH,P,Q,R,y_sig)

    K=P*JH'*1/(JH*P*JH'+R);
    M=F+K*(y_sig-H);
    P=(eye(size(Q))-K*JH)*P;  

    
end
    
    