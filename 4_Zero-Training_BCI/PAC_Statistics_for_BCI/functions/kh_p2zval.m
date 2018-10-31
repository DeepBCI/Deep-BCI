function zval = kh_p2zval(pval)
% zval() returns z-score from p-value
%
% input: p-value
% output: z-score
zval = abs(norminv(pval / 2));
end