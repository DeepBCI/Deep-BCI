%% Segmenation  -300ms ~~ 1200ms
function [normalBrake target] = normalBrake_Segmentation_car_new(normal,target,cnt,ival)

ival_f = -cnt.fs*(ival(1)/1000);
ival_r = cnt.fs*(ival(2)/1000);
%------------------------------Calcuation of startpoint,endpoint----------%

ind = min(find((normal + ival(1,1)) > 0));


m=0;
for i=ind:length(normal)
    m=m+1;

    startpoint(m) = normal(i) - ival_f ;
    endpoint(m) = normal(i) + ival_r;

end

%------------------------Construction of segmentation matrix--------------%
for i=1:size(cnt.x,2)
    for n=1:length( startpoint)
        normalBrake(:,i, n) = cnt.x(startpoint(n):endpoint(n),i);
    end
end


normalBrake = permute(normalBrake, [1 3 2]);
target.normalBrake = normalBrake;

end







