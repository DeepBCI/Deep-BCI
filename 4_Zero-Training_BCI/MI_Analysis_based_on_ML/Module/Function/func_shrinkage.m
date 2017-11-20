function [Cstar, gamma, T] = clsutil_shrinkage(X, varargin)
%CLSUTIL_SHRINKAGE - Shrinkage of covariance matrix with 'optimal' parameter
%
%Synopsis:
%  [CSTAR, GAMMA, T]= clsutil_shrinkage(X, OPT);
%
%Arguments:
%  X: DOUBLE [NxM] - Data matrix, with N feature dimensions, and M training points/examples.
%   OPT: PROPLIST - Structure or property/value list of optional properties: 
%    'Target': CHAR (default 'B') - Target mode of shrinkage, cf. [2],
%      'A' (identity matrix),
%      'B' (diagonal, common variance): default
%      'C' (common covariance)
%      'D' (diagonal, unequal variance)
%    'Gamma': DOUBLE - Shrinkage parameter. May be used to set the shrinkage parameter explicitly.
%    'Verbose': BOOL (default 0) - If set, verbose mode is activated
%
%Returns:
%  CSTAR: estimated covariance matrix
%  GAMMA: selected shrinkage parameter
%  T: target matrix for shrinkage
%
%Description:
%  Shrinkage of covariance matrix with 'optimal' parameter, see:
%
%  [1] Ledoit O. and Wolf M. (2004) "A well-conditioned estimator for
%  large-dimensional covariance matrices", Journal of Multivariate
%  Analysis, Volume 88, Number 2, February 2004 , pp. 365-411(47)
%
%  [2] Sch�fer, Juliane and Strimmer, Korbinian (2005) "A Shrinkage
%  Approach to Large-Scale Covariance Matrix Estimation and
%  Implications for Functional Genomics," Statistical Applications in
%  Genetics and Molecular Biology: Vol. 4 : Iss. 1, Article 32.
%
%Examples:
%  clsutil_shrinkage(X);
%
%See also:
%  TRAIN_RLDASHRINK

% 12-08-2010: adapted fast computation procedure for all four target
%             matrices: schultze-kraft@tu-berlin.de
% 12-09-2012: revised to fit with new naming standards and automatic
% opt-type checking (Michael Tangermann)


props= {'Target'     'B'     'CHAR(A B C D)'
        'Gamma'      'auto'  'CHAR(auto)|!DOUBLE'
        'Verbose'    0       'BOOL'};
%% add
opt.Target='B';
opt.Gamma='auto';
opt.Verbose=0;
%%

if nargin==0,
  Cstar= props;
  return
end
% 
% opt= opt_proplistToStruct(varargin{:});
% [opt, isdefault]= opt_setDefaults(opt, props,1);

if isequal(opt.Gamma, 'auto'),
  gamma = NaN;
elseif isreal(opt.Gamma),
  gamma = opt.Gamma;
else
  error('value for OPT.Gamma not understood');
end
  
%%% Empirical covariance
[p, n] = size(X);
Xn     = X - repmat(mean(X,2), [1 n]);
S      = Xn*Xn';
Xn2    = Xn.^2;

%%% Define target matrix for shrinkage
idxdiag    = 1:p+1:p*p;
idxnondiag = setdiff(1:p*p, idxdiag,'legacy');
switch(upper(opt.Target))
    case 'A'
        T = eye(p,p);
    case 'B'
        nu = mean(S(idxdiag));
        T  = nu*eye(p,p);
    case 'C'
        nu = mean(S(idxdiag));
        c  = mean(S(idxnondiag));
        T  = c*ones(p,p) + (nu-c)*eye(p,p);
    case 'D'
        T = diag(S(idxdiag));
    otherwise
        error('unknown value for OPT.Target')
end

%%% Calculate optimal gamma for given target matrix
if isnan(gamma)
    
    V     = 1/(n-1) * (Xn2 * Xn2' - S.^2/n);
    gamma = n * sum(sum(V)) / sum(sum((S - T).^2));
    
    %%% Handle special cases
    if gamma>1,
        if opt.Verbose,
            warning('gamma forced to 1');
        end
        gamma= 1;
    elseif gamma<0,
        if opt.Verbose,
            warning('gamma forced to 0');
        end
        gamma= 0;
    end
    
end

%%% Estimate covariance matrix
Cstar = (gamma*T + (1-gamma)*S ) / (n-1);
