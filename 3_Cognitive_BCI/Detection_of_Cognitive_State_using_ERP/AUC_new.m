function auc= AUC(label, out, varargin)
%[roc, auc, roc_x, hp]= roc_curve(label, out, <opt>)
%
% ROC - receiver operator characteristics
% only defined for 2-class problems. class 1 is the one to be detected,
% i.e., TP are samples of class 1 classified as '1' (out<0),
% while FP are samples of class 2 classified as '1'.
%
% IN   label  - true class labels, can also be a data structure (like epo)
%               including label field '.y'
%      out    - classifier output (as given, e.g., by the third output
%               argument of xvalidation)
%               size is [nShuffles nSamples]. when nShuffles is >1,
%               the resulting curve is the average of the ROC curves
%               of each shuffle.
%      opt 
%      .plot      - plot ROC curve, default when no output argument is given.
%      .linestyle - cellarray of linestyle property pairs,
%                   default {'linewith',2}.
%      .xlabel    - label of the x-axis, default 'false positive rate'.
%      .ylabel    - label of the y-axis, default 'true positive rate'.
%
% OUT  roc   - ROC curve
%      auc   - area under the ROC curve
%      roc_x - x-axis for plotting the ROC curve
%      hp    - handle to the plot
%
% SEE  xvalidation

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'plot', nargout==0, ...
                  'xlabel', 'false positive rate', ...
                  'ylabel', 'true positive rate', ...
                  'linestyle', {'linewidth',2});

if isstruct(label),
  label= label.y;
end

if size(label,1)>2,
  error('roc works only for 2-class problems');
end
N= sum(label, 2);

%%resort the samples such that class 2 comes first.
%%this makes ties count against the classifier, otherwise
%%roc_curve(y, ones(1,size(y,2))) could result in an auc>0.
[so,si]= sort([1 -1]*label);
label= label(:,si);
out= out(:,si);

nShuffles= size(out,1);
ROC= zeros(nShuffles, N(2));
for ii= 1:nShuffles,
  [so,si]= sort(out(ii,:));
  lo= label(:,si);
  idx2= find(lo(2,:));
  ncl1= cumsum(lo(1,:));
  ROC(ii,:)= ncl1(idx2)/N(1);
end
roc= mean(ROC, 1);

%% area under the roc curve
auc= sum(roc)/N(2);
roc_x= linspace(0, 1, N(2));

if opt.plot,
  hp= plot(roc_x, roc, opt.linestyle{:});
  xlabel(opt.xlabel);
  ylabel(opt.ylabel);
  title(sprintf('area under curve= %.4f', auc));
  axis([0 1 0 1], 'square');
end

