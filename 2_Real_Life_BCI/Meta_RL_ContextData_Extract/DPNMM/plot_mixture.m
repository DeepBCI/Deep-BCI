function plot_mixture(data,class_id,color)
% function plot_mixture(data,class_id,color)
%
% plots a mixture model given data (Dim x # points), class_id
% (vector), and a vector of colors.  The defaults color vector is
%
% color = {'k.', 'r.', 'g.', 'b.', 'm.', 'c.'};
%
% and if more than 6 classes are present, all classes with higher class_id
% are plotted as 'k.' unless more color arguments are given.  Additionally, 
% only the first two or three components of
% the training data are plotted
%
% this function should be proceeded by a clf 

[D,N] = size(data);

if D>3
    warning('Only the first three dimensions of the data will be plotted.');
    data = data(1:3,:);
end

if nargin < 3
        color = {'k.', 'r.', 'g.', 'b.', 'm.', 'c.'};
end


ucids = unique(class_id);
hold_state_modified = 0;
old_hold_state = get(gca,'NextPlot');
if ~strcmp(old_hold_state,'add')
    hold on;
    hold_state_modified = 1;
end

for i = 1:length(ucids)
    hits = class_id==ucids(i);
    if ucids(i)<length(color)
        if D == 2
            plot(data(1,hits),...
                data(2,hits),color{ucids(i)})
        elseif D==3
            plot3(data(1,hits),...
                data(2,hits),data(3,hits),color{ucids(i)});
        else
            error('Only two and three dimensional points allowed');
        end
    else
        if D==2
            plot(data(1,hits),...
                data(2,hits),'k.')
        elseif D==3
            plot3(data(1,hits),...
                data(2,hits),data(3,hits),'k.');
        else
            error('Only two and three dimensional points allowed');
        end
    end
end

if hold_state_modified 
    hold off;
end