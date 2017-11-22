function C = train_RLDAshrink(xTr, yTr, varargin)
% TRAIN_RLDASHRINK - Regularized LDA with automatic shrinkage selection
%
%Synopsis:
%   C = train_RLDAshrink(XTR, YTR)
%   C = train_RLDAshrink(XTR, YTR, OPTS)
%
%Arguments:
%   XTR: DOUBLE [NxM] - Data matrix, with N feature dimensions, and M training points/examples. 
%   YTR: INT [CxM] - Class membership labels of points in X_TR. C by M matrix of training
%                     labels, with C representing the number of classes and M the number of training examples/points.
%                     Y_TR(i,j)==1 if the point j belongs to class i.
%   OPT: PROPLIST - Structure or property/value list of optional
%                   properties. Options are also passed to clsutil_shrinkage.
%     'ExcludeInfs' - BOOL (default 0): If true, training data points with value 'inf' are excluded from XTR
%     'Prior' - DOUBLE (default ones(nClasses, 1)/nClasses): Empirical class priors
%     'StorePrior' - BOOL (default 0): If true, the prior will be stored with the classifier in C.prior
%     'Scaling' - BOOL (default 0): scale projection vector such that the distance between
%        the projected means becomes 2. Scaling only implemented for 2 classes so far. Using Scaling=1 will disable the use of a prior.
%     'StoreMeans' - BOOL (default 0): If true, the classwise means of the feature vectors
%        are stored in the classifier structure C. This can be used, e.g., for bbci_adaptation_pmean
%     'UsePcov' - BOOL (default 0): If true, the pooled covariance matrix is used instead of the average classwise covariance matrix.
%     'StoreCov',  - BOOL (default 0): If true, the covariance matrix will be stored with the classifier in C.cov
%     'StoreInvcov' - BOOL (default 0): If true, the inverse of the covariance matrix is stored in
%        the classifier structure C. This can be used, e.g., for bbci_adaptation_pcovmean
%     'StoreExtinvcov' - BOOL (default 0): If true, the extended inverse of the covariance will be stored with the classifier in C.extinvcov
%
%Returns:
%   C: STRUCT - Trained classifier structure, with the hyperplane given by
%               fields C.w and C.b.  C includes the fields:
%    'w' : weight matrix
%    'b' : FLOAT bias
%    'prior' : (optional) classwise priors
%    'means' :  (optional) classwise means
%    'cov' :  (optional) covariance matrix
%    'invcov' :  (optional) inverse of the covariance matrix
%    'extinvcov' : (optional) extended inverse of the covariance matrix
%
%Description:
%   TRAIN_RLDA trains a regularized LDA classifier on data X with class
%   labels given in LABELS. The shrinkage parameter is selected by the
%   function clsutil_shrinkage.
%
%   References: J.H. Friedman, Regularized Discriminant Analysis, Journal
%   of the Americal Statistical Association, vol.84(405), 1989. The
%   method implemented here is Friedman's method with LAMBDA==1. The
%   original RDA method is implemented in TRAIN_RDAREJECT.
%
%Examples:
%   train_RLDA(X, labels)
%   train_RLDA(X, labels, 'Target', 'D')
%   
%See also:
%   APPLY_SEPARATINGHYPERPLANE, CLSUTIL_SHRINKAGE, 
%   TRAIN_LDA, TRAIN_RDAREJECT

% Benjamin Blankertz
% 12-09-2012: revised to fit with new naming standards and automatic
% opt-type checking (Michael Tangermann)



if size(yTr,1)==1, yTr= [yTr<0; yTr>0]; end
nClasses= size(yTr,1);
opt.ExcludeInfs=0;
opt.Prior=0;
opt.UsePcov=0;
opt.StorePrior=0;
opt.StoreMeans=0;
opt.StoreCov=0;
opt.StoreInvcov=0;
opt.StoreExtinvcov=0;
opt.Scaling=0;
% props= {'ExcludeInfs'      0                             'BOOL'
%         'Prior'            ones(nClasses, 1)/nClasses    'DOUBLE[- 1]'
%         'UsePcov'          0                             'BOOL'
%         'StorePrior'       0                             'BOOL'
%         'StoreMeans'       0                             'BOOL'
%         'StoreCov'         0                             'BOOL'
%         'StoreInvcov'      0                             'BOOL'
% 		'StoreExtinvcov'   0                             'BOOL'
%         'Scaling'          0                             'BOOL'
%        };
% 
%    
% % get props list of the subfunction
% props_shrinkage= clsutil_shrinkage;
% 
% if nargin==0,
%   C= opt_catProps(props, props_shrinkage); 
%   return
% end
% 
% opt= opt_proplistToStruct(varargin{:});
% [opt, isdefault]= opt_setDefaults(opt, props);
% opt_checkProplist(opt, props, props_shrinkage);

% empirical class priors as an option (I leave 1/nClasses as default, haufe)
opt.Prior=ones(nClasses, 1)/nClasses;
if isnan(opt.Prior)
  opt.Prior = sum(yTr, 2)/sum(sum(yTr));
end

if opt.ExcludeInfs,
  ind = find(sum(abs(xTr),1)==inf);
  xTr(:,ind) = [];
  yTr(:,ind) = [];
end

d= size(xTr, 1);
X= zeros(d,0);
C_mean= zeros(d, nClasses);
for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean(:,ci)= mean(xTr(:,idx),2);
  if ~opt.UsePcov,
    X= [X, xTr(:,idx) - C_mean(:,ci)*ones(1,length(idx))];
  end
end

%changes
% opt_shrinkage= opt_substruct(opt, props_shrinkage(:,1));
% if opt.UsePcov,
%   [C_cov, C.gamma]= clsutil_shrinkage(xTr, opt_shrinkage);
% else
%   [C_cov, C.gamma]= clsutil_shrinkage(X, opt_shrinkage);
% end
[C_cov, C.gamma]= clsutil_shrinkage(X);

C_invcov= pinv(C_cov);

C.w= C_invcov*C_mean;
C.b= -0.5*sum(C_mean.*C.w,1)' + log(opt.Prior);

if nClasses==2
  C.w= C.w(:,2) - C.w(:,1);
  C.b= C.b(2)-C.b(1);
end

if opt.Scaling,
  if nClasses>2,
    error('Scaling only implemented for 2 classes so far (TODO!)');
  end
  if ~isdefault.Prior,
    warning('prior ignored, when scaling (TODO!)');
  end
  C.w= C.w/(C.w'*diff(C_mean, 1, 2))*2;
  C.b= -C.w' * mean(C_mean,2);
end

if opt.StorePrior,
  C.prior= opt.Prior;
end
if opt.StoreMeans,
  C.mean= C_mean;
end
if opt.StoreCov,
  C.cov= C_cov;
end
if opt.StoreInvcov,
  C.invcov= C_invcov;
end
if opt.StoreExtinvcov,
  % pooled(!) covariance
  feat= [ones(1,size(xTr,2)); xTr];
  C.extinvcov= inv(feat*feat'/size(xTr,2));
  % Alternative (with shrinkage):
%  [C_extpcov, C_gamma_extpcov]= ...
%       clsutil_shrinkage([ones(1,size(xTr,2)); xTr], opt_shrinkage);
%  C.extinvcov= pinv(C_extpcov);
% But this subtracts the pooled mean, which seems not to be appropriate
% ?? Ask Carmen, when she's back ??
end
