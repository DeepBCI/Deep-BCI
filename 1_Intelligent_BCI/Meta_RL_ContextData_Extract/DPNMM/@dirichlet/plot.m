function plot(d)

if length(d.alpha) > 2
    error('plotting not supported for Dirichlet distributions greater than 2D')
end

X = linspace(0,1,500)';
Y = 1-X;

data = [X Y];

Z = pdf(d,data);
scatter3(X,Y,Z);