%% Segmenation  -300ms ~~ 1200ms
function target = targetSegmentation_car_new(stimulus,nonstimulus,cnt,ival,ival1)

ival_f = -cnt.fs*(ival(1)/1000);
ival_r = cnt.fs*(ival(2)/1000);
ival1 = cnt.fs*(ival1/1000);

%------------------------------Calcuation of startpoint,endpoint----------%
% a = size(stimulus.NontargetBrakeOn_stim,2);
% b = size(stimulus.NontargetLongBrakeOn_stim,2);
% a = exist('a');
% b = exist('b');

m=0;
for i=1:length(stimulus.TargetBrake_stim)
    m=m+1;
%         startpoint(m) = stimulus.TargetBrake_stim(i) - 60;
%         endpoint(m) = stimulus.TargetBrake_stim(i) + 240;
    startpoint(m) = stimulus.TargetBrake_stim(i) - ival_f ;
    endpoint(m) = stimulus.TargetBrake_stim(i) + ival_r;
%      startpoint(m) = stimulus.TargetBrake_stim(i) - 400;
%     endpoint(m) = stimulus.TargetBrake_stim(i) + 700;
end

m=0;
% if a>=1
    for i=1:length(stimulus.NontargetBrakeOn_stim)
        m=m+1;
        %         startpoint(m) = stimulus.NontargetBrake_stim(i) - 60;
        %         endpoint(m) = stimulus.NontargetBrake_stim(i) + 240;
        startpoint1(m) = stimulus.NontargetBrakeOn_stim(i) - ival_f ;
        endpoint1(m) = stimulus.NontargetBrakeOn_stim(i) + ival_r;
        %      startpoint(m) = stimulus.NontargetBrake_stim(i) - 400;
        %     endpoint(m) = stimulus.NontargetBrake_stim(i) + 700;
    end
% end

m=0;
% if b>=1
    for i=1:length(stimulus.NontargetLongBrakeOn_stim)
        m=m+1;
        %         startpoint(m) = stimulus.NontargetLongBrake_stim(i) - 60;
        %         endpoint(m) = stimulus.NontargetLongBrake_stim(i) + 240;
        startpoint2(m) = stimulus.NontargetLongBrakeOn_stim(i) - ival_f ;
        endpoint2(m) = stimulus.NontargetLongBrakeOn_stim(i) + ival_r;
        %      startpoint(m) = stimulus.NontargetLongBrake_stim(i) - 400;
        %     endpoint(m) = stimulus.NontargetLongBrake_stim(i) + 700;
    end
% end

m=0;
for i=1:length(stimulus.Right_stim)
    m=m+1;
%     startpoint1(m) = stimulus.Right_stim(i) - 60;
%     endpoint1(m) = stimulus.Right_stim(i) + 240;
    startpoint3(m) = stimulus.Right_stim(i) - ival_f;
    endpoint3(m) = stimulus.Right_stim(i) + ival_r;
%      startpoint1(m) = stimulus.Right_stim(i) - 400;
%     endpoint1(m) = stimulus.Right_stim(i) + 700;
end

m=0;
for i=1:length(stimulus.Left_stim)
    m=m+1;
%     startpoint1(m) = stimulus.Left_stim(i) - 60;
%     endpoint1(m) = stimulus.Left_stim(i) + 240;
    startpoint4(m) = stimulus.Left_stim(i) - ival_f;
    endpoint4(m) = stimulus.Left_stim(i) + ival_r;
%      startpoint1(m) = stimulus.Left_stim(i) - 400;
%     endpoint1(m) = stimulus.Left_stim(i) + 700;
end

m=0;
for i=1:length(stimulus.Human_stim)
    m=m+1;
%     startpoint2(m) = stimulus.Human_stim(i) - 60;
%     endpoint2(m) = stimulus.Human_stim(i) + 240;
    startpoint5(m) = stimulus.Human_stim(i) - ival_f;
    endpoint5(m) = stimulus.Human_stim(i) + ival_r;
%         startpoint2(m) = stimulus.Human_stim(i) - 400;
%     endpoint2(m) = stimulus.Human_stim(i) + 700;
end
%-------------------------------------------------------------------------%
%------------------------Construction of segmentation matrix--------------%
for i=1:size(cnt.x,2)
    for k=1:size(nonstimulus.refresh_stim,2)
        for n=1:length( stimulus.TargetBrake_stim)
            if (nonstimulus.refresh_stim(k) - ival1 > stimulus.TargetBrake_stim(n)) || ...
                    (nonstimulus.refresh_stim(k) + ival1 < stimulus.TargetBrake_stim(n))
                target.target_TargetBrake(:,i, n) = cnt.x(startpoint(n):endpoint(n),i);
                if (nonstimulus.refresh_stim(k) +ival1 < stimulus.TargetBrake_stim(n)) && (k < size(nonstimulus.refresh_stim,2)) 
                    k=k+1;
                end
            end
        end
%         if a>=1
            for n=1:length( stimulus.NontargetBrakeOn_stim)
                if (nonstimulus.refresh_stim(k) - ival1 > stimulus.NontargetBrakeOn_stim(n)) || ...
                        (nonstimulus.refresh_stim(k) + ival1 < stimulus.NontargetBrakeOn_stim(n))
                    target.target_NontargetBrakeOn(:,i, n) = cnt.x(startpoint1(n):endpoint1(n),i);
                    if (nonstimulus.refresh_stim(k) +ival1 < stimulus.NontargetBrakeOn_stim(n)) && (k < size(nonstimulus.refresh_stim,2))
                        k=k+1;
                    end
                end
            end
%         end
%         if b>=1
            for n=1:length( stimulus.NontargetLongBrakeOn_stim)
                if (nonstimulus.refresh_stim(k) - ival1 > stimulus.NontargetLongBrakeOn_stim(n)) || ...
                        (nonstimulus.refresh_stim(k) + ival1 < stimulus.NontargetLongBrakeOn_stim(n))
                    target.target_NontargetLongBrakeOn(:,i, n) = cnt.x(startpoint2(n):endpoint2(n),i);
                    if (nonstimulus.refresh_stim(k) +ival1< stimulus.NontargetLongBrakeOn_stim(n)) && (k < size(nonstimulus.refresh_stim,2))
                        k=k+1;
                    end
                end
            end
%         end
        for n=1:length( stimulus.Right_stim)
            if (nonstimulus.refresh_stim(k) - ival1 > stimulus.Right_stim(n)) || ...
                    (nonstimulus.refresh_stim(k) + ival1 < stimulus.Right_stim(n))
                target.target_Right(:,i, n) = cnt.x(startpoint3(n):endpoint3(n),i);
                if (nonstimulus.refresh_stim(k) +ival1 < stimulus.Right_stim(n)) && (k < size(nonstimulus.refresh_stim,2)) 
                    k=k+1;
                end
            end
        end
        
        for n=1:length( stimulus.Left_stim)
            if (nonstimulus.refresh_stim(k) - ival1 > stimulus.Left_stim(n)) || ...
                    (nonstimulus.refresh_stim(k) + ival1 < stimulus.Left_stim(n))
                target.target_Left(:,i, n) = cnt.x(startpoint4(n):endpoint4(n),i);
                if (nonstimulus.refresh_stim(k) +ival1 < stimulus.Left_stim(n)) && (k < size(nonstimulus.refresh_stim,2)) 
                    k=k+1;
                end
            end
        end
        
        for n=1:length( stimulus.Human_stim)
            if (nonstimulus.refresh_stim(k) - ival1 > stimulus.Human_stim(n)) || ...
                    (nonstimulus.refresh_stim(k) + ival1 < stimulus.Human_stim(n))
                target.target_Human(:,i, n) = cnt.x(startpoint5(n):endpoint5(n),i);
                if (nonstimulus.refresh_stim(k) +ival1 < stimulus.Human_stim(n)) && (k < size(nonstimulus.refresh_stim,2)) 
                    k=k+1;
                end
            end
        end
    end
end
%-------------------------------------------------------------------------%
%------------------------Baseline correction------------------------------%
% for i=1:size(cnt.x,2)
%     for n=1:length(stimulus.TargetBrake_stim)
%         Mean(i, n) = mean(target.target_brake(1:21, i, n));
%         target.target_brake(:, i, n) = target.target_brake(:,i, n)-Mean(i, n);
%     end
% 
%     for n=1:length(stimulus.Left_stim)
%         Mean1(i, n) = mean(target.target_right(1:21, i, n));
%         target.target_right(:, i, n) = target.target_right(:,i, n)-Mean1(i, n);
%     end
% 
%     for n=1:length(stimulus.Human_stim)
%         Mean2(i, n) = mean(target.target_human(1:21, i, n));
%         target.target_human(:, i, n) = target.target_human(:,i, n)-Mean2(i, n);
%     end
% 
% end

target.target_TargetBrake = permute(target.target_TargetBrake, [1 3 2]);
% if a>=1
    target.target_NontargetBrakeOn = permute(target.target_NontargetBrakeOn, [1 3 2]);
% end
% if b>=1
    target.target_NontargetLongBrakeOn = permute(target.target_NontargetLongBrakeOn, [1 3 2]);
% end
target.target_Right = permute(target.target_Right, [1 3 2]);

target.target_Left = permute(target.target_Left, [1 3 2]);

target.target_Human = permute(target.target_Human, [1 3 2]);

% target.count_target_NontargetBrakeOn = a;

% target.count_target_NontargetLongBrakeOn = b;


end






