function [Xwh, mu, invMat, whMat] = whiten(X,epsilon)
%function [X,mu,invMat] = whiten(X,epsilon)
%
% ZCA whitening of a data matrix (make the covariance matrix an identity matrix)
%
% WARNING
% This form of whitening performs poorly if the number of dimensions are
% much greater than the number of instances
%
%
% INPUT
% X: rows are the instances, columns are the features
% epsilon: small number to compensate for nearly 0 eigenvalue [DEFAULT =
% 0.0001]
%
% OUTPUT
% Xwh: whitened data, rows are instances, columns are features
% mu: mean of each feature of the orginal data
% invMat: the inverse data whitening matrix
% whMat: the whitening matrix
%
% EXAMPLE
%
% X = rand(100,20); % 100 instance with 20 features
% 
% figure;
% imagesc(cov(X)); colorbar; title('original covariance matrix');
% 
% [Xwh, mu, invMat, whMat] = whiten(X,0.0001);
% 
% figure;
% imagesc(cov(Xwh)); colorbar; title('whitened covariance matrix');
% 
% Xwh2 = (X-repmat(mean(X), size(X,1),1))*whMat; 
% figure;
% plot(sum((Xwh-Xwh2).^2),'-rx'); title('reconstructed whitening error (should be 0)');
% 
% Xrec = Xwh*invMat + repmat(mu, size(X,1),1);
% figure;
% plot(sum((X-Xrec).^2),'-rx'); ylim([-1,1]); title('reconstructed data error (should be zero)');
%
% Author: Colorado Reed colorado-reed@uiowa.edu


if ~exist('epsilon','var')
    epsilon = 0.0001;
end

mu = mean(X); 
X = bsxfun(@minus, X, mu);
A = X'*X;
[V,D,notused] = svd(A);
whMat = sqrt(size(X,1)-1)*V*sqrtm(inv(D + eye(size(D))*epsilon))*V';
Xwh = X*whMat;  
invMat = pinv(whMat);

end

