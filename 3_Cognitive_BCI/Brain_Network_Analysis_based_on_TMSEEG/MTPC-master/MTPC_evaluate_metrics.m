function [stats,config] =  MTPC_evaluate_metrics(GT,design,config)


% This function implements the multi-threshold permutation correction test
% for graph theory measurements. The function will
% allow the testing of an array of GT metrics using any design that can be
% fitted with a standard GLM.
%
% See Drakesmith et al (2015) 10.1016/j.neuroimage.2015.05.011
%
% (C) Mark Drakesmith, Cardiff University
%
% Usage:
%
%   [stats,config] =  MTPC_evaluate_metrics(GT,design)
% or
%   [stats,config] =  MTPC_evaluate_metrics(GT,design,config)
%
% INPUTS
%
% GT is a cell array of GT metrics (on cell per petric). in each cell is an
%         array with dimensions (n_subj x n_thresh x [other dimensions of the GT
%             metric (e.g. n_nodes)].
% design:  This is the design matrix of for statistical analysis, including any
%               covariates. The size of the matrix is n_subj x n_var.
%
% config: This is a structure containing various parameters, details below:
%
%  terms:   This is a matrix of terms to test. the size of this matrix is
%             n_var x n_terms.
%             Some examples follow:
%            *  For a correlation with only one variable specified in the
%               design matrix, use terms=1.
%            *  For a simple t-test between two groups, where design is a
%               binary classification fro group membership, do terms=[1]
%            *  To test for any effect of the 1st variable, while controlling
%               for a 2nd variable, do terms=[1 0];
%            *  To test for main effects and interaction effect of two
%               variables, do: terms=[1 0 ; 0 1 ; 1 1 ];
%            *  To test for an interaction effects betwen two vairables,
%               while controlling for a third variable, do: terms=[1 1 0];
%          WARNING - signs in the terms matrix will be ignored. Because the
%          F statistic is used, resultant p-values will reflect a
%          significant effect in either direction (equivalent to a 2-tailed
%          test). Direction of the effect can be determined from the sign
%          of the t-value. However explicit 1-tailed tests are not
%          supported.
%
% rand:     if rand is a scalar value, this indicate the number of
%           randomisations to perform. If rand is a n_subj x n_rand matrix of
%         randomisation indices (this can be used to do the same randomisations
%          previously done).
%          WARNING: exchangeability blocks (e.g. for repeated-measures designs)
%                             are not yet supported!!
%
% thresh: A 1xn_thresh vector of threshold values, or a n_subjxn_thresh
%              vector of subject specific threshold values. default =
%              1:n_thresh;
%              WARNING: Although not essential, it is advisable to
%              specify the thresholding, especially if the threshold
%              vector is non-monotonic as this influences the computation
%              of the AUC.
%
% method:   Which method to implement:
%            'mtpc+scauc' Performed multi-threshold permutation correction,
%                           with additional super-critical AUC test (default).
%            'mtpc'       Performs multi-threshold permutation correction only.
%            'scauc'      Performs the super-critical AUC test, without
%                            multi-threshold correction.
%            'auc'        Uses  the standard AUC approach.
%            Note that even though 'scauc' and 'auc' wont correct statistics across
%            thresholds, it will still correct across any dimensions
%            specfied in 'corr_dims'.
%
% modeltype:  Specfied the kind of statistical model used;
%              'glm'      Uses the GLM approach (equivilent to t-test,
%                           ANOVA, ANCOVA, Pearson correlation, etc) (default)
%              'ranksum'  Uses the rank-sum test between groups (aka
%                          Mann-Whitney U test).
%              'spearman'  Uses the Spearman correlation
%                         WARNING: This implimnetation assumes one critical
%                         value for both positive and negative corretions,
%                         which may not be valid. Use with cauton!           '
%
% alpha_corr:         This is the significance level applied when correcting across thresholds
%                            or other dimensions (default = 0.05).
%
% corr_dims:   A vector of  dimensions of the GT metric (in addition to threshold) to be
%              corrected across. (e.g for a non-specific GT metrics), additional
%              correction for multiple comparisons across nodes can be carried
%              out. corr_dims can also be a  cell-array of vectors, to a specific
%              specific set of dimensions for each GT metric.
%              corr_dims can also be 'all' or 'none'. (default = 'all').
%
% varnames:    An a cell array of strings indicating the name of each
%              variable in the design matrix (default =
%              {'X1','X2',...'Xn_var'}).
%
% keep_glm:    [0 or 1], specifies if, the the full GLM and ANOVA results for each test
%              should be kept (note this will only be done for the unrandomised tests).
%              WARNING: This can return a very large cell-array if turned on!).
%              default = 0;
% keep_rand:  [0 or 1]. Specified if the randomisation indices should be
%              kept. This is useful if one wishes to correct new data using the same
%             randomisations. (default = 0). The config field 'rand' will be
%             replaced with the  randomisation indices
% output_rand:    [0 or 1]. specifies if output should include the
%                      randomisation indices.
% smooth      Specifes if smoothing should be applied prior to computing
%               AUC. Only applicable if method='auc' (defualt=0);
% smooth_fwhm      Specifes the FWHM of smoothing kernel if applied to AUCs
%                  (default is the agerage derivative of the threshold vector)
% parallel:  [0 or 1]. Tries to enable parallel processing if parallel
%                 toolbox is installed (default = 1);
%
%
% OUTPUTS
%  stats = a 1xn_metrics structure array with various stats outputs
%              (depended on which config options have been specified).
%
% The following fields have size n_terms x [other GT dimensions]
%
% p_orig2, t_orig2 and f_orig2  are the original, t, F and p values of the
%                              initial GLM fit at the highest peak.
% p_corr2       is the permutation corrected p-value at the highest peak.
% f_crit           is the critical value of F after correction
% sig_corr2     Indicates if the effect is significant after correction.
% max_t_thresh_idx  The index of the threshold vector corresponding to the
%                    maximum F statistic.
% GT_maxt       A cell array of the GT values for each test where the peak
%                        effect is seen (useful for plotting).
% GT_auc        A cell array of the GT AUC values for each test.
% scauc_corr    The super-critical  AUC of effect
% scauc_crit    The super-critical AUC of the randomisation effect
% scauc_sig     Indicates if the effect is significant with super-critical AUC.
% glm               A cell array of GLM structures for each fit
% tbl               A cell array of ANOVA tables for each fit.
% coefnames     Cell array of coefficient names (based on varnames)
% design        The design matrix
% terms         The terms matrix
%
%
% The following fields have size n_terms x n_thresh x [other GT dimensions]
%
%  p_orig1, t_orig1 and f_orig1  are the original, t, F and p vlaues of the
%                              GLM fit for all thresholds
%  p_corr1:       The corrected p-values for all thresholds
%  sig_corr1:     Binary array indicating if the result if significant at
%                 all thresholds


% create empty config if not present.
if ~exist('config','var')
    config=[];
end


n_subj=size(GT{1},1);
n_thresh=size(GT{1},2);
n_metric=length(GT);
n_vars=size(design,2);

% set defaults
if ~isfield(config,'rand');       config.rand=1000;                  end
if ~isfield(config,'method');     config.method='mtpc+scauc';       end
if ~isfield(config,'test_scauc'); config.test_scauc=1;              end
if ~isfield(config,'alpha_glm');  config.alpha_glm=0.05;            end
if ~isfield(config,'alpha_corr'); config.alpha_corr=0.05;           end
if ~isfield(config,'terms');      config.terms = [(eye(n_vars))];   end
if ~isfield(config,'corr_dims');  config.corr_dims = 'all' ;        end
if ~isfield(config,'keep_glm');   config.keep_glm = 0 ;             end
if ~isfield(config,'keep_rand');  config.keep_rand = 0 ;            end
if ~isfield(config,'parallel');   config.parallel = 1 ;             end
if ~isfield(config,'thresh');     config.thresh = 1:n_thresh;       end
if ~isfield(config,'smooth');     config.smooth = 0;                end
if ~isfield(config,'smooth_fwhm');config.smooth_fwhm = mean(diff(config.thresh));       end
if ~isfield(config,'modeltype');     config.modeltype = 'glm';                end
%
% if ~isfield(config,'alpha_scauc');
%     if strcmp(config.method,'scauc')
%         config.alpha_scauc=config.alpha_glm;
%     else
%         config.alpha_scauc=0.5;
%     end
% end


% generate default varnames if not specfied.
if ~isfield(config,'varnames');
    for i=1:n_vars
        config.varnames{i} = ['X' num2str(i)];
    end
end

% add response to variable names if not added.
if length(config.varnames)==n_vars
    config.varnames{end+1}='Y';
end

% check if final varnames is expected length
if length(config.varnames)~=(1+n_vars)
    error('Variable names do not match number of variables!');
end

% check conditions are OK for modeltype

if strcmp(config.modeltype,'glm');
    modeltype=1;
elseif strcmp(config.modeltype,'ranksum');
    modeltype=2;
    if size(design,2)>1 || length(unique(design))>2
        error('Only a single-column 2-group design is supported when using ranksum');
    end
elseif strcmp(config.modeltype,'spearman');
    modeltype=3;
    if length(unique(design))>2
        error('Only a design of a single column of values is support when using Spearman');
    end
    warning('Spearman method is not properly implimented. Use with caution!');
end

% turn on parallelisation if available
if config.parallel
    warning off
    %     try
    %         matlabpool close force
    %     catch e
    %         throw(e)
    %     end
    
    try
        matlabpool('open', feature('numCores'));
    catch e
        e
        
        fprintf('WARNING: Unable to launch matlabpool. Error message follows:\n');
        fprintf([e.identifier '\n']);
        fprintf([e.message '\n']);
    end
    warning on
end


% check 'rand' is specfied correctly
if numel(config.rand)>1 & ~isequal(size(config.rand,1),n_subj);
    error('if ''rand'' is a matrix of randomisation indices, it should have the same number of rows as the number of subjects');
elseif numel(config.rand)>1
    % if randomisation indices are specifed, simply append with unrandomised vector
    rand_idx=config.rand;
    n_rand=size(rand_idx,2);
    rand_idx=[[1:n_subj]' rand_idx];
    
else
    % if randomisation indices are not specifed, simply append with unrandomised vector
    n_rand=config.rand;
    for i=1:n_rand+1
        if i==1
            rand_idx=[1:n_subj]';
        else
            [rubbish rand_idx(:,i)]=sort(rand(1,n_subj));
        end
    end
end

% get terms matrix and signs of terms

% terms_temp=(config.terms);
% terms_temp(find(terms_temp==0))=1;
% termsigns=prod(sign(terms_temp),2);

% make sure all terms are valid (i.e no empty rows or repeated rows)
terms=abs(config.terms);
term_remove=find(sum(terms,2)==0);
if ~isempty(term_remove)
    warning('Removing empty rows from terms matrix');
end
n_terms=size(terms,1);

for i=1:n_terms
    for j=i+1:n_terms
        if isequal(terms(i,:),terms(j,:));
            warning('Removing duplicate row from terms matrix');
            term_remove(end+1)=j;
        end
    end
end

terms(term_remove,:)=[];
n_terms=size(terms,1);



alpha_glm=config.alpha_glm;
alpha_corr=config.alpha_corr;

% set up corr_dims
corr_dims=config.corr_dims;
if ~iscell(corr_dims)
    % if not specfied as a corr_dim for each metrix, copy across a n_metric
    % cell array.
    corr_dims=repmat({corr_dims},1,n_metric);
end


% create high-res threshold vector using interpretation
thresh=config.thresh;
thresh_interp=linspace(thresh(1),thresh(end),2000);

% check method is valid
if ~(strcmp(config.method,'mtpc+scauc') | ...
        strcmp(config.method,'mtpc') | ...
        strcmp(config.method,'scauc') |...
        strcmp(config.method,'auc') | ...
        strcmp(config.method,'smthauc'));
    error('Unknown method. Valid options are ''mtpc+scauc'', ''mtpc'', ''scauc'' and ''auc''.')
end

stats=[];

for m=1:n_metric
    
    % if specified to do the AUC method, compute AUCs for GT metric first
    % (this collapses GT metrics across threhsolds;
    if strcmp(config.method,'auc')
        % create high-res threshold vector using interpretation (and rest
        % thresh to original value).
        thresh=config.thresh;
        thresh_interp=linspace(thresh(1),thresh(end),2000);
        
        % permute GT matrix so threshodls are first dimension
        gt_temp=permute(GT{m},[2 1 3 4 5 6 7 8]);
        
        % interpolate the gt metrics tohigh res
        gt_interp=interp1(thresh,gt_temp,thresh_interp);
        
        % this adds smoothing to thresholded GT metrics if specfied
        if config.smooth
            
            % convert fwlm to stdst
            sd=config.smooth_fwhm./2.335;
            
            kernel=normpdf(thresh_interp-median(thresh_interp),0,sd);
            
            kernel=kernel.*(max(gt_interp(:,1))-min(gt_interp(:,1)))+min(gt_interp(:,1));
            kernel=kernel.*(max(kernel)-min(kernel))+min(kernel);
            
            % Construct blurring window.
            
            % constrcut kernel. Gasuian kernals full widths should be twice the FWHM
            windowWidth = int16(2.*config.smooth_fwhm./mean(diff(thresh_interp)));
            halfwidth=windowWidth/2;
            kernel=gausswin(double(windowWidth));
            kernel = kernel / sum(kernel); % Normalize.
            
            for j=1:prod(size(gt_interp(1,:)))
                % pad the edges of the GT data
                gt_interp_temp=[ones(1,windowWidth/2).*gt_interp(1,j) gt_interp(:,j)' ones(1,windowWidth/2).*gt_interp(end,j)];
                
                % convolve GT data with kernel
                gt_smooth=conv(gt_interp_temp,kernel);
                
                l=length(gt_interp(:,j));
                start_idx=windowWidth;
                end_idx=start_idx+l-1;
                
                
                % replace gt_interp with the smoothed version
                gt_interp(:,j)=gt_smooth(start_idx:end_idx);
                
            end
            
            
        end
        
        % compute AUC of interpolated GT metric
        gt_auc=trapz(thresh_interp,gt_interp);
        % permute back to original dimensions
        GTm=permute(gt_auc,[2 1 3 4 5 6 7 8]);
        
        n_thresh=1;
        thresh=1;
        
    else
        GTm=GT{m};
    end
    
    % interprete which diemsniosn to apply correction across
    
    if isequal(corr_dims{m},'all')
        corr_dims{m}=2:ndims(GT{m});
    elseif isequal(corr_dims{m},'none')
        corr_dims{m}=2;
    end
    
    % dont correct across thresholds when method is 'scauc' or 'auc'
    if strcmp(config.method,'auc') || strcmp(config.method,'scauc')
        corr_dims{m}(find(corr_dims{m}==2))=[];
    end
    
    
    
    for i=1:n_rand+1
        
        if i==1
            
            % create empty arrays for storing randomised p-values
            %      t_all_temp=zeros([n_terms size(GTm(1,:,:,:,:,:,:))]);
            p_all_temp=zeros([n_terms size(GTm(1,:,:,:,:,:,:))]);
            t_all_temp=zeros([n_terms size(GTm(1,:,:,:,:,:,:))]);
            
            f_all_temp=zeros([n_terms size(GTm(1,:,:,:,:,:,:))]);
            
            if config.keep_glm
                stats(m).glm=cell(size(GTm(1,:,:,:,:,:,:)));
                stats(m).tbl=cell(size(GTm(1,:,:,:,:,:,:)));
            end
            
        end
        
        
        for th=1:n_thresh
            fprintf('Analysing GT metric %u, randomisation %u, threshold %u. \n',m,i-1,th);
            
            if modeltype==1
                
                parfor j=1:numel(GTm(1,1,:))
                    [ p_all_temp2(:,j) t_all_temp2(:,j) f_all_temp2(:,j) coefnames{j} mdl{j} tbl{j}] = ...
                        glm_subfunc(GTm(rand_idx(:,i),th,j),design,terms,config.varnames);
                end
            elseif modeltype==2
                parfor j=1:numel(GTm(1,1,:))
                    [ p_all_temp2(:,j) f_all_temp2(:,j)] =  ranksum_subfunc(GTm(rand_idx(:,i),th,j),design);
                    t_all_temp2(:,j) = f_all_temp2(:,j);
                end
            elseif modeltype==3
                parfor j=1:numel(GTm(1,1,:))
                    [ p_all_temp2(:,j) f_all_temp2(:,j)] = ...
                        spearman_subfunc(GTm(rand_idx(:,i),th,j),design);
                    t_all_temp2(:,j) = f_all_temp2(:,j);
                end
            end
            
            %                 %
            %                 % fit the glm
            %                 mdl=LinearModel.fit(design,GTm(rand_idx(:,i),th,j),config.terms,'VarNames',config.varnames);
            %
            %                 %   end
            %                 %              if i==1 & config.keep_glm
            %                 %                  glm{c,th,j}=mdl{j};
            %                 %              end
            %
            %                 %    for j=1:numel(GTm(1,1,:))
            %                 for c=1:n_terms
            %                     if termsigns(c)>0
            %                         p_all_temp(c,i,th,j)=mdl.Coefficients{c,4};
            %                         t_all_temp(c,i,th,j)=mdl.Coefficients{c,3};
            %                     else
            %                         p_all_temp(c,i,th,j)=1-mdl.Coefficients{c,4};
            %                         t_all_temp(c,i,th,j)=-mdl.Coefficients{c,3};
            %                     end
            %                 end
            
            
            p_all_temp(:,i,th,:)=permute(p_all_temp2,[1 3 4 2]);
            t_all_temp(:,i,th,:)=permute(t_all_temp2,[1 3 4 2]);
            f_all_temp(:,i,th,:)=permute(f_all_temp2,[1 3 4 2]);
            
            if config.keep_glm
                stats(m).glm(th,:)=mdl;
                stats(m).tbl(th,:)=tbl;
            end
            
            
            
        end
    end
    fprintf('Completed all randomisations for GT metric %u. Computed corrected p-values\n',m);
    
    
    % do the MPTC
    
    % find the min and max of the t-stats across thresholds (and dimensions for
    % subsiquent repmat).
    f_max_temp=f_all_temp(:,2:end,:,:,:,:,:,:);
    f_max_rep=ones(1,ndims(p_all_temp));
    %     t_min_temp=t_all_temp(:,2:end,:,:,:,:,:,:);
    %     t_min_rep=ones(1,ndims(p_all_temp));
    
    t_orig_rep=ones(1,ndims(p_all_temp));
    t_orig_rep(2)=n_rand;
    
    
    for d=1:length(corr_dims{m})
        f_max_temp=max(f_max_temp,[],corr_dims{m}(d)+1);
        f_max_rep(corr_dims{m}(d)+1)=size(t_all_temp,corr_dims{m}(d)+1);
        %         t_min_temp=min(t_min_temp,[],corr_dims{m}(d)+1);
        %         t_min_rep(corr_dims{m}(d)+1)=size(t_all_temp,corr_dims{m}(d)+1);
    end
    
    
    
    
    % repmat the f_crit and f_orig so their dimensions match
    f_max_temp=repmat(f_max_temp(:,:,:,:,:,:,:,:,:),f_max_rep);
    %     t_min_temp=repmat(t_min_temp(:,:,:,:,:,:,:,:,:),t_min_rep);
    t_orig_temp=repmat(t_all_temp(:,1,:,:,:,:,:,:,:),t_orig_rep);
    p_orig_temp=repmat(p_all_temp(:,1,:,:,:,:,:,:,:),t_orig_rep);
    f_orig_temp=repmat(f_all_temp(:,1,:,:,:,:,:,:,:),t_orig_rep);
    
    
    
    
    
    % p_corr is the the proportion of randomisations where t_max>t_orig
    p_corr_temp=mean(double(f_max_temp>f_orig_temp),2);
    
    % find the corresponding critical F-statistic
    f_crit_temp=prctile(f_max_temp,100*(1-alpha_corr),2);
    
    %     % p_corr is the the proportion of randomisations where t_min<t_orig and
    %     % where p_orig is greater than alpha_glm
    %     p_corr_min_temp=mean(double(t_min_temp<t_orig_temp & p_orig_temp>alpha_glm),2);
    %
    %     % combine the two signs of p_corr so that this is effectlivly 2-tailed
    %     p_corr_max_temp(find(p_corr_min_temp-p_corr_max_temp)>0)=p_corr_min_temp(find(p_corr_min_temp-p_corr_max_temp)>0);
    %  p_corr_temp=min(cat(10,p_corr_min_temp,p_corr_max_temp),[],10);
    
    % inserts p_corr and p_orig into output cell array
    % squeeze, so final dims are n_terms x n_thresh x [other GT
    % dims];
    stats(m).p_orig1=permute(p_all_temp(:,1,:,:,:,:,:,:,:),[1 3 4 5 6 7 8 9 2]);
    stats(m).t_orig1=permute(t_all_temp(:,1,:,:,:,:,:,:,:),[1 3 4 5 6 7 8 9 2]);
    stats(m).f_orig1=permute(f_all_temp(:,1,:,:,:,:,:,:,:),[1 3 4 5 6 7 8 9 2]);
    
    stats(m).f_crit=permute(f_crit_temp,[1 3 4 5 6 7 8 9 2]);
    stats(m).p_corr1=permute(p_corr_temp,[1 3 4 5 6 7 8 9 2]);
    stats(m).sig_corr1=stats(m).p_corr1<alpha_corr;
    
    if modeltype==1
        stats(m).coefnames=coefnames{1};
        stats(m).design=design;
        stats(m).terms=terms;
    end
    
    % if doing MTPC, we need to find the peak effect across thresholds
    if strcmp(config.method,'mtpc') | strcmp(config.method,'mtpc+scauc') | strcmp(config.method,'scauc')
        % get the threshold of maximum effect
        [mat_temp tidx_corr2]=max(stats(m).t_orig1,[],2); % 여기가 nanmax였는데 max로 바꿈
        stats(m).max_t_thresh_idx=permute(tidx_corr2,[1 3 4 5 6 7 8 9 2]);
        
        %
        stats(m).p_orig2=zeros(size(stats(m).max_t_thresh_idx));
        stats(m).t_orig2=stats(m).p_orig2;
        stats(m).p_corr2=stats(m).p_orig2;
        stats(m).sig_corr2=stats(m).p_orig2;
        for c=1:n_terms
            for j=1:numel(GTm(1,1,:))
                stats(m).p_orig2(c,j)=stats(m).p_orig1(c,tidx_corr2(c,1,j),j);
                stats(m).t_orig2(c,j)=stats(m).t_orig1(c,tidx_corr2(c,1,j),j);
                stats(m).p_corr2(c,j)=stats(m).p_corr1(c,tidx_corr2(c,1,j),j);
                stats(m).sig_corr2(c,j)=stats(m).sig_corr1(c,tidx_corr2(c,1,j),j);
                
                stats(m).GT_maxt{c,j}=GTm(:,tidx_corr2(c,1,j),j);
            end
        end
    elseif strcmp(config.method,'auc')
        % if doing AUC, store the GT AUCs
        for c=1:n_terms
            for j=1:numel(GTm(1,1,:))
                stats(m).GT_auc{c,j}=GTm(:,1,j);
            end
        end
    end
    
    
    % if specfied, do the super-crtiical AUC test.
    
    
    if strcmp(config.method,'mtpc+scauc') | strcmp(config.method,'scauc')
            
        if all(stats(m).p_corr1(:)>alpha_corr(:))
            % skip if none of the p-values are significant
            continue
        end
                
        fprintf('Testing super-critical AUCs\n');
           
              
        % permute all t-values )original and rand) so thresholds are firstr
        % dimension
        
        f_temp=permute(f_all_temp,[3 1 2 4 5 6 7 8]);
        %interpolate the orgiinal t values and the noisy t-values
        f_interp=interp1(thresh,f_temp,thresh_interp);
        
        % get crtitical t values in right shape
        f_crit_temp=repmat(permute(f_max_temp(:,1,1,:,:,:,:,:),[3 1 2 4 5 6 7 8]),[2000 1 n_rand+1 1 1 1 1 1 1]);
        
        
        % threshold t values at at critical t
        f_interp = f_interp - f_crit_temp;
        f_interp(find(f_interp<0))=0;
        f_auc=trapz(thresh_interp,f_interp);
        
        
        % get the percentile
        % compute estimated rand AUC
        f_auc_rand=f_auc(:,:,2:end,:,:,:,:,:);
        f_auc_rand(find(f_auc_rand<0))=NaN;
        %  f_auc_crit=prctile(f_auc_rand,100.*(1-alpha_scauc),3);
        f_auc_rand=nanmean(f_auc_rand,3);
        
        
        % store results
        stats(m).scauc_corr=squeeze(f_auc(:,:,1,:,:,:,:,:));
        stats(m).scauc_crit=squeeze(f_auc_rand);
        stats(m).scauc_sig=stats(m).scauc_corr>stats(m).scauc_crit;
        
    end
    
end


if config.keep_rand
    config.rand=rand_idx;
end

config.terms=terms;


function [p_temp t_temp f_temp coefnames mdl tbl] =  glm_subfunc(gt,design,terms,varnames)

n_terms=size(terms,1);

% add intercept term
terms = [zeros(1,size(terms,2)) ; terms];
%
% fit the glm
mdl=LinearModel.fit(design,gt,terms,'VarNames',varnames);

% compute F statistics from GLM
tbl=anova(mdl);

% get coefnames
coefnames=mdl.CoefficientNames(2:end);



for c=1:n_terms
    
    f_temp(c)=tbl{c,4};
    t_temp(c)=mdl.Coefficients{c+1,3};
    p_temp(c)=tbl{c,5};
    
end


function [p_temp f_temp] =  ranksum_subfunc(gt,design)

% binariise design

design_unique=unique(design);
design(find(design_unique(1)))=0;
design(find(design_unique(2)))=1;

% correlate GT data with tat int he design matrix

[p_temp,h,stats] = ranksum(gt(find(design)),gt(find(~design)));
f_temp=stats.ranksum;
%



function [p_temp f_temp] =  spearman_subfunc(gt,design)


% correlate GT data with tat int he design matrix

[f_temp,p_temp] = corr(gt,design,'type','Spearman');

%
f_temp=abs(f_temp(1,2));
p_temp=p_temp(1,2);





