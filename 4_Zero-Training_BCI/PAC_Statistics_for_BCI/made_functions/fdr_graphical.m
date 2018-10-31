function [psort, id, crit_id, crit_pval] = fdr_graphical(pval, qlevel)
psort = [];
id = [];
crit_id = [];
crit_pval = [];


if nargin < 2
    qlevel = 0.05;
end
[psort, id] = sort(pval);
m_alpha = qlevel;
m = length(psort);
k = 1:m;

y = (m_alpha / m) * k;

figure, plot(k, psort);
hold on; plot(k, y);
xlim([1 32])
xlabel('k', 'fontsize', 13); ylabel('sorted p-val', 'fontsize', 13);

sig_id = find(psort <= y);

disp(sig_id);
if ~isempty(sig_id)
    crit_id = id(sig_id);
    crit_pval = pval(crit_id);
    plot([sig_id(end) sig_id(end)], get(gca, 'ylim'), 'k--');
    annotation('textbox',[.65 .815 .1 .1],'string',sprintf('whole = %02d, k = %02d', 32, length(sig_id)));
end

end