function y = kh_PCA(data, elements)
% input:
%
%   data: [time x ch]
%   element: # of pca component to be used
%
% output:
%
%   y: [ ch x # of component ]

cov_erp = cov(data);
[W, D] = eig(cov_erp); % D: elgenvalue matrix;
% ascending order (eigenvalue - diagonal)
% choose the components

W_PCA = W(:, (end+1)-elements:end);
W_SUM = sum(W_PCA, 1);

% portion of significance
for i=1:elements
    WP_PCA(:, i) = W_PCA(:, i) / W_SUM(i);
end

y = WP_PCA;

end