function [ normal ] = normalBrake_list_new( cnt , response )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
j=1;k=1;l=1;m=1;n=1;o=1;p=1;q=1;
flag = 0;
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (i - response.TargetBrake_stim(j) > 1000 || i - response.TargetBrake_stim(j) < -1000) && ...
                (i - response.NontargetBrakeOn_stim(m) > 1000 || i - response.NontargetBrakeOn_stim(m) < -1000) && ...
                (i - response.NontargetLongBrakeOn_stim(n) > 1000 || i - response.NontargetLongBrakeOn_stim(n) < -1000) && ...
                (i - response.Right_stim(o) > 1000 || i - response.Right_stim(o) < -1000) && ...
                (i - response.Left_stim(p) > 1000 || i - response.Left_stim(p) < -1000) && ...
                (i - response.Human_stim(q) > 1000 || i - response.Human_stim(q) < -1000)
            
            brake(1,k) = i;
            flag = 1;
            k=k+1;
        end
    elseif (cnt.x(i,end) < 1*(10^-3)) && flag == 1
        flag = 0;
        normal(1,l) = min(brake);
        l=l+1;
        k=1;
        clear brake;
    end
    
    if j < size(response.TargetBrake_stim,2)
        if i == response.TargetBrake_stim(j) + 1000
            j=j+1;
        end
    end
    if m < size(response.NontargetBrakeOn_stim,2)
        if i == response.NontargetBrakeOn_stim(m) + 1000
            m=m+1;
        end
    end
    if n < size(response.NontargetLongBrakeOn_stim,2)
        if i == response.NontargetLongBrakeOn_stim(n) + 1000
            n=n+1;
        end
    end
    
    if o < size(response.Right_stim,2)
        if i == response.Right_stim(o) + 1000
            o=o+1;
        end
    end
    
    if p < size(response.Left_stim,2)
        if i == response.Left_stim(p) + 1000
            p=p+1;
        end
    end
    
    if q < size(response.Human_stim,2)
        if i == response.Human_stim(q) + 1000
            q=q+1;
        end
    end
    
    
end

end




