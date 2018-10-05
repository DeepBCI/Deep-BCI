function [ response, restimulus ]  = response_list_new( cnt, stimulus )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% TargetBrake
j=1;k=1;l=1;n=1;o=1;p=1;q=1;r=1;s=1;t=1;u=1;v=1;w=1;x=1;
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.TargetBrake_stim(j) - i < 100) && (stimulus.TargetBrake_stim(j) - i >= 0)
            j = j+1;
        end
        if (i - stimulus.TargetBrake_stim(j) <= 180) && (i - stimulus.TargetBrake_stim(j) > 100)
            response.TargetBrake_stim(1,u) = i;
            restimulus.TargetBrake_stim(1,u) = stimulus.TargetBrake_stim(j);
            j = j+1;
            u = u+1;
        elseif (i - stimulus.TargetBrake_stim(j) > 180)
            j = j+1;
        end
        if (i - stimulus.TargetBrake_stim(j) <= 100) && (i - stimulus.TargetBrake_stim(j) > 0)
            j = j+1;
        end
        if j >= size(stimulus.TargetBrake_stim,2)
            break;
        end
        
    end
end




%% NontargetBrake
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.NontargetBrake_stim(k) - i < 100) && (stimulus.NontargetBrake_stim(k) - i >= 0)
            k = k+1;
        end
        if (i - stimulus.NontargetBrake_stim(k) <= 240) && (i - stimulus.NontargetBrake_stim(k) > 100)
            response.NontargetBrakeOn_stim(1,q) = i;
            restimulus.NontargetBrakeOn_stim(1,q) = stimulus.NontargetBrake_stim(k);
            k = k+1;
            q= q+1;
        elseif i - stimulus.NontargetBrake_stim(k) > 600
            restimulus.NontargetBrakeOff_stim(1,r) = stimulus.NontargetBrake_stim(k);
            k = k+1;
            r= r+1;
        end
        if (i - stimulus.NontargetBrake_stim(k) > 240) && (i - stimulus.NontargetBrake_stim(k) <= 600)
            k = k+1;
        end
        if (i - stimulus.NontargetBrake_stim(k) <= 100) && (i - stimulus.NontargetBrake_stim(k) > 0)
            k = k+1;
        end
        
        if k >= size(stimulus.NontargetBrake_stim,2)
            break;
        end
        
    end
end



%% NontargetLongBrake
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.NontargetLongBrake_stim(l) - i < 100) && (stimulus.NontargetLongBrake_stim(l) - i >= 0)
            l = l+1;
        end
        if (i - stimulus.NontargetLongBrake_stim(l) <= 240) && (i - stimulus.NontargetLongBrake_stim(l) > 100)
            response.NontargetLongBrakeOn_stim(1,s) = i;
            restimulus.NontargetLongBrakeOn_stim(1,s) = stimulus.NontargetLongBrake_stim(l);
            l = l+1;
            s = s+1;
        elseif i - stimulus.NontargetLongBrake_stim(l) > 600
            restimulus.NontargetLongBrakeOff_stim(1,t) = stimulus.NontargetLongBrake_stim(l);
            l = l+1;
            t = t+1;
        end
        if (i - stimulus.NontargetLongBrake_stim(l) > 240) && (i - stimulus.NontargetLongBrake_stim(l) <= 600)
            l = l+1;
        end
        
        if (i - stimulus.NontargetLongBrake_stim(l) <= 100) && (i - stimulus.NontargetLongBrake_stim(l) > 0)
            l = l+1;
        end
        
        
        if l >= size(stimulus.NontargetLongBrake_stim,2)
            break;
        end
        
    end
end




%% Right
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.Right_stim(n) - i < 100) && (stimulus.Right_stim(n) - i >= 0)
            n = n+1;
        end
        if (i - stimulus.Right_stim(n) <= 180) && (i - stimulus.Right_stim(n) > 100)
            response.Right_stim(1,v) = i;
            restimulus.Right_stim(1,v) = stimulus.Right_stim(n);
            n = n+1;
            v = v+1;
        elseif i - stimulus.Right_stim(n) > 180
            n = n+1;
        end
        if (i - stimulus.Right_stim(n) <= 100) && (i - stimulus.Right_stim(n) > 0)
            n = n+1;
        end
        
        if n >= size(stimulus.Right_stim,2)
            break;
        end
        
    end
end

%% Left
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.Left_stim(o) - i < 100) && (stimulus.Left_stim(o) - i > 0)
            o = o+1;
        end
        if (i - stimulus.Left_stim(o) <= 180) && (i - stimulus.Left_stim(o) > 100)
            response.Left_stim(1,w) = i;
            restimulus.Left_stim(1,w) = stimulus.Left_stim(o);
            o = o+1;
            w = w+1;
        elseif i - stimulus.Left_stim(o) > 180
            o = o+1;
        end
        
        if (i - stimulus.Left_stim(o) <= 100) && (i - stimulus.Left_stim(o) > 0)
            o = o+1;
        end
        
        if o >= size(stimulus.Left_stim,2)
            break;
        end
        
    end
end



%% Human
for i=1:size(cnt.x,1)
    if cnt.x(i,end) > 1*(10^-3)
        if (stimulus.Human_stim(p) - i < 100) && (stimulus.Human_stim(p) - i > 0)
            p = p+1;
        end
        if (i - stimulus.Human_stim(p) <= 180) && (i - stimulus.Human_stim(p) > 100)
            response.Human_stim(1,x) = i;
            restimulus.Human_stim(1,x) = stimulus.Human_stim(p);
            p = p + 1;
            x = x+1;
        elseif i - stimulus.Human_stim(p) > 180
            p = p + 1;
        end
        
        if (i - stimulus.Human_stim(p) <= 100) && (i - stimulus.Human_stim(p) > 0)
            p = p+1;
        end
        
        if p >= size(stimulus.Human_stim,2)
            break;
        end
        
    end
end



end

