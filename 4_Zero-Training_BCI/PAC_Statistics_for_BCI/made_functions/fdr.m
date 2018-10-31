function varargout = fdr(varargin)
% Computes the FDR-threshold for a vector of p-values.
%
% Usage:
% [pthr,pcor,padj] = fdr(pvals)
%                    fdr(pval,q)
%                    fdr(pval,q,cV)
%
% Inputs:
% pvals  = Vector of p-values.
% q      = Allowed proportion of false positives (q-value).
%          Default = 0.05.
% cV     = If 0, uses an harmonic sum for c(V). Otherwise uses c(V)=1.
%          Default = 1.
%
% Outputs:
% pthr   = FDR threshold.
% pcor   = FDR corrected p-values.
% padj   = FDR adjusted p-values.
%
% Note that the corrected and adjusted p-values do **not** depend
% on the supplied q-value, but they do depend on the choice of c(V).
%
% References:
% * Benjamini & Hochberg. Controlling the false discovery
%   rate: a practical and powerful approach to multiple testing.
%   J. R. Statist. Soc. B (1995) 57(1):289-300.
% * Yekutieli & Benjamini. Resampling-based false discovery rate
%   controlling multiple test procedures for multiple testing
%   procedures. J. Stat. Plan. Inf. (1999) 82:171-96.
%
% ________________________________
% Anderson M. Winkler
% Research Imaging Center/UTHSCSA
% Dec/2007 (first version)
% Nov/2012 (this version)
% http://brainder.org

% Accept arguments
switch nargin,
    case 0,
        error('Error: Not enough arguments.');
    case 1,
        pval = varargin{1};
        qval = 0.05;
        cV   = 1;
    case 2,
        pval = varargin{1};
        qval = varargin{2};
        cV   = 1;
    case 3,
        pval = varargin{1};
        qval = varargin{2};
        if varargin{3}, cV = 1;
        else cV = sum(1./(1:numel(pval))) ;
        end
    otherwise
        error('Error: Too many arguments.')
end

% Check if pval is a vector
if numel(pval) ~= length(pval),
    error('p-values should be a row or column vector, not an array.')
end

% Check if pvals are within the interval
if min(pval) < 0 || max(pval) > 1,
    error('Values out of range (0-1).')
end

% Check if qval is within the interval
if qval < 0 || qval > 1,
    error('q-value out of range (0-1).')
end

% ========[PART 1: FDR THRESHOLD]========================================

% Sort p-values
[pval,oidx] = sort(pval);

% Number of observations
V = numel(pval);

% Order (indices), in the same size as the pvalues
idx = reshape(1:V,size(pval));

% Line to be used as cutoff
thrline = idx*qval/(V*cV);

% Find the largest pval, still under the line
thr = max(pval(pval<=thrline));

% Deal with the case when all the points under the line
% are equal to zero, and other points are above the line
if thr == 0,
    thr = max(thrline(pval<=thrline));
end

% Case when it does not cross
if isempty(thr), thr = 0; end

% Returns the result
varargout{1} = thr;

% ========[PART 2: FDR CORRECTED]========================================

if nargout == 2 || nargout == 3,
    
    % p-corrected
    pcor = pval.*V.*cV./idx;

    % Sort back to the original order and output
    [~,oidxR] = sort(oidx);
    varargout{2} = pcor(oidxR);
end

% ========[PART 3: FDR ADJUSTED ]========================================

if nargout == 3,

    % Loop over each sorted original p-value
    padj = zeros(size(pval));
    prev = 1;
    for i = V:-1:1,
        % The p-adjusted for the current p-value is the smallest slope among
        % all the slopes of each of the p-values larger than the current one
        % Yekutieli & Benjamini (1999), equation #3.
        padj(i) = min(prev,pval(i)*V*cV/i);
        prev = padj(i);
    end
    varargout{3} = padj(oidxR);
end

% That's it!