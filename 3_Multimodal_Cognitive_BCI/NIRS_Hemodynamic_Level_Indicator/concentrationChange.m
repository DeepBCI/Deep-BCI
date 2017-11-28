function [total]=concentrationChange(HbO_HbR)

for step = 1:length(HbO_HbR(:,1))
hbo(step) = sum(HbO_HbR(1:step,1));
end


for step = 1:length(HbO_HbR(:,1))
hbr(step) = sum(HbO_HbR(1:step,2));
end

hbo = hbo';
hbr = hbr';


total = [hbo,hbr];

end
