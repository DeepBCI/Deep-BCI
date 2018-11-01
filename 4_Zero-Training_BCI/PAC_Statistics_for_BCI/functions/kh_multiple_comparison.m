%% kh_multiple_comparision,m
%
% It
%
% Usage
%   inputs:
%          dat: data structure that contains information
%             dat.val: data input1 [observation x dat1 x dat2]
%             dat.val2: data input2 (in case of 'same' and 'diff')
%
%             dat.permap: permutation distribution for data dimension
%
%          alpha: significance level
%          approach:
%           'bonferrnoi', 'FDR', 'cluster-tval', 'cluster-size', 'min-max'
%          CAUTION: cluter based correction only works for 2D data
%
%          group: 'none', 'same', 'diff'
%            'none': no condition difference (resting state, ...)
%            'same': same group, different condition (baseline/post-stimulus)
%            'diff': group difference (control vs. experiment)
%          CAUTION: for 'same' and 'diff', we use ttest based multiple comparison
%          correction (ttest for same and ttest2 for diff)
%
%   outputs:
%          corrected val
%
function [corrected, critical] = kh_multiple_comparison(dat, alpha, approach, group)

switch group
    case {'same', 'diff'}
        [corrected, critical] = multiple_comparison_ttest(dat, alpha, approach, group);
    case 'none'
        [corrected, critical] = multiple_comparison(dat, alpha, approach);
end

end

% ------------------------------------------------ sub functions

% ------------------- normal multiple comparision
function [corrected, critical] = multiple_comparison()

% simple multiple comparision correction methods

end

% ------------------- ttest based multiple comparision
function [corrected, critical] = multiple_comparison_ttest(dat, alpha, approach, group)

% ------ 1. ttest without correction between conditions(groups)
% user can modify ttest setting (e.g. tail) accroding to the data
switch group
    case 'same'
        [H, P, CI, STATS] = ttest(dat.val, dat.val2, ...
            'alpha', alpha, 'tail', 'both', 'dim', 1);
    case 'diff'
        [H, P, CI, STATS] = ttest2(dat.val, dat.val2, ...
            'alpha', alpha, 'tail', 'both', 'dim', 1);
end
raw_tmap = squeeze(STATS.tstat);
raw_critical = squeeze(H);
figure, pcolor(raw_tmap); shading interp;  hold on;
contour(raw_critical, 1, '--', 'Color', 'k', 'linewidth', 3);
title('t-value map without correction');
colormap('jet');

% masking of raw_tmap and raw_critical for later use
mask_raw_tmap = raw_tmap;
mask_raw_critical = raw_critical;

mask_corrected_tmap = mask_raw_tmap;
mask_corrected_tmap(mask_raw_critical==0) = 0;

% ------ 2. simple multiple comparision correction methods
if strcmp(approach, 'bonferrnoi')
    test_dim = size(dat.val, 2) * size(dat.val, 3);
    threshold = alpha / test_dim;
    switch group
        case 'same'
            [H, P, CI, STATS] = ttest(dat.val, dat.val2, ...
                'alpha', threshold, 'tail', 'both', 'dim', 1);
        case 'diff'
            [H, P, CI, STATS] = ttest(dat.val, dat.val2, ...
                'alpha', threshold, 'tail', 'both', 'dim', 1);
    end
    corrected_tmap = squeeze(STATS.tstat);
    corrected_critical = squeeze(H);
    figure, pcolor(raw_tmap); shading interp;  hold on;
    contour(corrected_critical, 1, '--', 'Color', 'k', 'linewidth', 3);
    title('t-value map with Bonferrnoi');
    colormap('jet');

    corrected = corrected_tmap;
    critical = corrected_critical;
    
    % ------3. complex multiple comparision correction methods
elseif ismember(approach, {'min-max', 'cluster-size', 'cluster-tval'})
    npermute = 1000;
    max_val = zeros(npermute, 2);
    max_cluster_size = zeros(npermute, 1);
    max_cluster_tval = zeros(npermute, 1);
    permap = cat(1, dat.val, dat.val2); % [observation (1+2) x dat1 x dat2]
    
    for nbpermute = 1:npermute
        disp([num2str(nbpermute) '/' num2str(1000) ' permutations...']);
        % shffling condition labels
        permute_id = randperm(size(permap, 1));
        tmp_permap = permap(permute_id, :, :);
        condition_length = size(dat.val, 1);
        conditionA = tmp_permap(1:condition_length, :, :);
        conditionB = tmp_permap(condition_length+1:end, :, :);

        switch group
            case 'same'
                [H, P, CI, STATS] = ttest(conditionA, conditionB, ...
                    'alpha', alpha, 'tail', 'both', 'dim', 1);
            case 'diff'
                [H, P, CI, STATS] = ttest2(conditionA, conditionB, ...
                    'alpha', alpha, 'tail', 'both', 'dim', 1);
        end
        tmap = squeeze(STATS.tstat); % 2D
        h_critical = squeeze(H);
        corrected_tmap = tmap;
        corrected_tmap(h_critical==0) = 0;
        
        % define cluster
        isCluster = bwconncomp(corrected_tmap); % 8 neighbor
        if numel(isCluster.PixelIdxList) > 0 % number of pixels that composite cluster
            % count sizes of clusters
            tmpClusterSize = cellfun(@length, isCluster.PixelIdxList);
            
            tmp_sum_tval = [];
            for i=1:numel(isCluster.PixelIdxList)
                tmp_sum_tval(i) = sum(corrected_tmap(isCluster.PixelIdxList{i}));
            end
            
            % store size of biggest cluster and sum of t-value
            max_cluster_size(nbpermute) = max(tmpClusterSize);
            max_cluster_sum_tval(nbpermute) = max((tmp_sum_tval)); % signed max
            min_cluster_sum_tval(nbpermute) = min((tmp_sum_tval));
            
        end
        
        % pixel based min-max value correction
        % get extreme values (min and max)
        sort_map = sort(reshape(tmap, 1, [])); % strech map
        max_val(nbpermute, :) = [min(sort_map), max(sort_map)];
    end
    
    if strcmp(approach, 'min-max')
        minmax_low_bound = prctile(max_val(:, 1), 100*alpha/2);
        minmax_high_bound = prctile(max_val(:, 2), 100-100*alpha/2);       
        
        thresholded = find(mask_raw_tmap > minmax_low_bound & mask_raw_tmap < minmax_high_bound);
        % mask_raw_tmap(thresholded) = 0;
        mask_raw_critical(thresholded) = 0;
        
        figure, pcolor(mask_raw_tmap); shading interp; hold on;
        contour(mask_raw_critical, 1, '--', 'Color', 'k', 'linewidth', 3);
        title('t-value map with min-max correction');
        colormap('jet');

        corrected = [];
        critical = mask_raw_critical;
        
    elseif strcmp(approach, 'cluster-size') || strcmp(approach, 'cluster-tval')
        cluster_size_bound = prctile(max_cluster_size, 100-(100*alpha));
        sum_tval_low_bound = prctile(min_cluster_sum_tval, (100*alpha/2));
        sum_tval_high_bound =prctile(max_cluster_sum_tval, 100-(100*alpha/2));
        
        isCluster_raw = bwconncomp(mask_corrected_tmap);
        for i=1:isCluster_raw.NumObjects
            % if real clusters are too small, remove them by setting to zero
            if strcmp(approach, 'cluster-tval')
                sum_of_t_values = sum((mask_corrected_tmap(isCluster_raw.PixelIdxList{i})));
                if sum_of_t_values > 0 && sum_of_t_values < sum_tval_high_bound
                    mask_raw_critical(isCluster_raw.PixelIdxList{i}) = 0;
                end
                
                if sum_of_t_values < 0 && sum_of_t_values > sum_tval_low_bound
                    mask_raw_critical(isCluster_raw.PixelIdxList{i}) = 0;
                end
            elseif strcmp(approach, 'cluster-size')
                if max(mask_raw_tmap(isCluster_raw.PixelIdxList{i})) < cluster_size_bound
                    mask_raw_critical(isCluster_raw.PixelIdxList{i}) = 0;
                end
            end
            
        end
        figure, pcolor(mask_raw_tmap); shading interp; hold on;
        contour(mask_raw_critical, 1, '--', 'Color', 'k', 'linewidth', 3);
        title('t-value map with cluster correction');
        colormap('jet');

        corrected = [];
        critical = mask_raw_critical;
    end
    
end

end
