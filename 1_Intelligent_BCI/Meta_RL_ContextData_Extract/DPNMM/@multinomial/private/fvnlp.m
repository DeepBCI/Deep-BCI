function [lp, md] = fvnlp(x,mu,covariance)
% fast_vectorized_normal_probability(points,mean,covariance_matrix)
%
% INPUT:
%   points - the points to evaluate (column vectors)
%   mean - mean of normal distribution (column vector of same length)
%   covariance - covariance matrix 
% OUTPUT:
%   p      - probability
%   md     - Mahalanobis distance

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

dim = length(x(:,1));
num_points = length(x(1,:));
[scale,inv_cov] =pcgf(covariance);

y=x-repmat(mu,1,num_points);

md2 = sum(y.*(inv_cov*y),1);
% md2 = zeros(1,num_points);
% for i = 1:num_points
%     md2(i) = y(:,i)'*inv_cov*y(:,i);
% end
md = (md2).^(.5);

lp = -md2/2 +log(scale);
% p = scale*exp(-md2/2);



function [scale,inverse] = pcgf(covariance)

dimension = length(covariance(:,1));

% SINGULAR
if(rcond(covariance)<eps)
        inverse = pinv(covariance,eps);
        scale = 1/((2*pi)^(dimension/2)*sqrt(det(covariance)));
%     inverse = inv(covariance);
%     inverse = eye(size(covariance));
%     scale = 0;
else
    inverse = inv(covariance);
    scale = 1/((2*pi)^(dimension/2)*sqrt(det(covariance)));
end
