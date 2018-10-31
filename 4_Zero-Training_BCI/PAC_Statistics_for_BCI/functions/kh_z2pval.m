function pval = kh_z2pval(zval)
% zval() returns p-value from z-score
%
% input: z-score
% output: p-value
pval = normcdf(-abs(zval), 0, 1) .* 2;
end