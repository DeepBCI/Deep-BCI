function hndls = plot(d,colors)
% GMM\PLOT

% Copyright October, 2006, Brown University, Providence, RI. 
% All Rights Reserved 

% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a commercial
% product is hereby granted without fee, provided that the above copyright
% notice appear in all copies and that both that copyright notice and this
% permission notice appear in supporting documentation, and that the name of
% Brown University not be used in advertising or publicity pertaining to
% distribution of the software without specific, written prior permission. 

% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE. IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY
% SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
% RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
% CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
% CONNECTION WITH THE USE.

% Author: Frank Wood fwood@cs.brown.edu

if(nargin < 2)
    colors = {'k','r','g','b','m','c'};
end


nc = d.mixture_model.num_components;
hndls = zeros(nc,1);
invp = 1.96;
hold on
for(i=1:nc)
    g = d.mixture_model(i);
    g = g{1};
    mu = g.mean;
    sigma = g.covariance;
    
    [vec, val] = eig(sigma(1:2,1:2));
    axes = invp*sqrt(svd(val));
%     angles = -atan2(vec(1,:),vec(2,:));
    
    t= linspace(0,2*pi);
    ellip = vec*invp*sqrt(val)*[cos(t);sin(t)] + repmat([mu(1); mu(2)],1,100);
    ellip = ellip';
    axes = axes';
    color_index = i;
    if(color_index>length(colors))
        color_index=length(colors);
    end
    hndls(i) = plot(ellip(:,1),ellip(:,2),colors{color_index});
end
hold off
