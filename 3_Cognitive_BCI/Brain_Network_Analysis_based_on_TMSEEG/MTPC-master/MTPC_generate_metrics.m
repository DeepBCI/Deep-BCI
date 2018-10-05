function [GT_data,config] =  MTPC_generate_metrics(CM,config)

% This function computes GT metrics for matrices across thresholds, in
% preperation for muilt-threhsold permutation correction.
%
% See Drakesmith et al (2015) 10.1016/j.neuroimage.2015.05.011
%
% (C) Mark Drakesmith, Cardiff University
%
% Usage:
%   [GT_data,config] =  MTPC_generate_metrics(CM,config)
%
% CM:     a nxnxsubj 3D array of connectiivty matrices for each subject
%            (subject concataneted across 3rd dimension). 
%         Althernatively, CM can be a 4D array where the fourth dimension corresponds to 
%         different thresholds (this is useful if using a non-standard
%         threhsolding stratagy not employed in here). 
% config:   A structure, specifying various optional paramaters. 
%            Detailed below:
%
% thresh  A 1xn_thresh vector of threshold values, or a n_subjxn_thresh
%          vector of subject specific threshold values. default =
%          linspace(0,median(CM),20); 
% thresh_type: Either 'absolute','relative', or 'density' (default =
%           'absolute'). 
% weights:  sepcfied if how weights in the thresholded matrices should be
%           treated. 'original', or 'binary'. Alternativly, this can be an array, same size as CM, 
%           which provide an alterantive connectivity matrices used for weights (e.g. if
%           one wishes to weight edges by myelin fraction,  but wants to use streamline count to threshold). 
% GT_funcs: This is a cell array of text, indicating what functions to evaluete to generate graph metrics.
%           The format should be:
%             '[GT argout1 .. argoun ] = func_name(CM,arg2,...,argn)'
%           where 'CM' corresponds to the argument to be used for the connectivity
%           matrix, and 'GT' is the output of the fucntion coresponding to the
%           function of interest. For example, using functions from BCT;
%              Global efficinecy, use 'GT = efficiency_wei(CM,1)' ;
%              Modularity with gamma=0.5, use '[~ GT]=modularity_und(CM,0.5)' ;
% keep_CMthresh: 0 or 1, specifes is output config structure should contain
%               the 4D array of thresholded CMs (default = 0).

%
% default is a set of basic GT metrics from the BCT toolbox;
%   
%
% OUTPUTS
% GT_data:  a n_metrics cell array of GT metrics. Each cel contains an
% array of size n_subj x n_thresh x additional dimensions given by the
% metric
% config:  final config used for generatign metrics. 

% initial sanity checks
if ~exist('config','var')
    config=[];
end

if ~exist('CM','var')
    error('Need connectivity matrix to work with');
end

CM_idx=find(CM>0 & ~isnan(CM));

% set config defaults

if ~isfield(config,'thresh_type');  config.thresh_type='absolute';                   end
if ~isfield(config,'weights');      config.weights='original';                       end
if ~isfield(config,'keep_CMthresh');      config.keep_CMthresh=0;                    end


% set defaults for thresholds depending on structure of CM
use_CMthresh=0;
if ndims(CM)==3
if ~isfield(config,'thresh');       config.thresh=linspace(0,median(CM(CM_idx)),50); end
elseif ndims(CM)==4
    fprintf('Using pre-thresholded matrices. Ignoring any values in ''thresh''\n');
    use_CMthresh=1;
    config.thresh=1:size(CM,4);
elseif ndims(CM)<3 || ndims(CM)>4 || size(CM,1)~=size(CM,2)
    error('CM is not in a recognised format.');
end
        
    
thresh=config.thresh;
n_thresh = size(thresh,2);
n_subj = size(CM,3);

       



% where do  weights for the graph come from?
if isnumeric(config.weights)

    if ~isequal(size(config.weights),size(CM(:,:,:,1)));
        error('Alternative weight-definitions must be a 3D array matching size of CM.');
    end
    if use_CMthresh
        if ndims(config.weights)==3
        CMw=repmat(config.weights,[1 1 1 n_thresh]);
        elseif ndims(config.weights)==4
            CMw=config.weights;
        end
    end
elseif isequal(config.weights,'original')
    CMw=CM;
elseif isequal(config.weights,'binary')
    CMw=logical(ones(size(CM)));
else
    error('Unknown option for ''weights''.');
    end



% if all of CMw is binary, force 'weights' to be binary
if isequal(unique(CMw(:)),[0 1])
    config.weights = 'binary';
end



% set default sets of BCT functions to use 
% WARNING: these defaults assume undirected connectivity!!!
if ~isfield(config,'GT_funcs');
    if isequal(config.weights,'binary')
        GT_funcs={...
            'GT = degrees_und(CM);',...
            'GT = efficiency_bin(CM,1);',...
            'GT = clustering_coef_bu(CM);',...
            'GT = betweenness_bin(CM);',...
            'D=distance_bin(CM); GT=charpath(D);'};
    else
        GT_funcs={...
            'GT = strengths_und(CM);',...
            'GT = efficiency_wei(CM,1);',...
            'GT = clustering_coef_wu(CM);',...
            'GT = betweenness_wei(CM);',...
            'D=distance_wei(1./CM); GT=charpath(D);'};
    end
else
    GT_funcs=config.GT_funcs;
end
n_metric = length(GT_funcs);


if ischar(GT_funcs)
    GT_funcs={GT_funcs};
end









if size(thresh,1)==1
    thresh=repmat(thresh,n_subj,1);
end

if isequal(config.thresh_type,'absolute')
    thresh_abs=thresh;
end



fprintf('Evaluting %u GT functions across %u thresholds for %u matrices... \n',n_metric,n_thresh,n_subj);

% go through each threshold
for s=1:n_subj
    
    fprintf('Processing matrix %u\n',s);
    
    if ~use_CMthresh % dont need to do this if pre-thresholded matrices are provided.
        
        if isequal(config.thresh_type,'density');
            
            % absolut threhsolds correspond to a cosntant density threhsold
            
            [~,thresh_abs(s,:)] = density_threshold_graph(CM(:,:,s),thresh(s,:));
        end
        
        % absolute thresholds correspond to a relative threshold (relative to
        % largest value in CM)
        if isequal(config.thresh_type,'relative');
            thresh_abs(s,:)=thresh(s,:).*max(max(CM(:,:,s)));
        end
    end
        

     % apply thresholds       
     for th=1:n_thresh
         
         
         if use_CMthresh
             % use thresholded matrices if provided
             CM_thresh_temp=CMw(:,:,s,th);
             CM_thresh_temp(find(CM(:,:,s,th)==0))=0;
         else
             % use the computed absolute threshold
             CM_thresh_temp=CMw(:,:,s);
             CM_thresh_temp(find(CM(:,:,s)<=thresh_abs(s,th)))=0;
         end
         
         for m=1:n_metric
             
                      if s==1 & th==1
                        GT_data{m} = [];
                      end
         
             
             %avlulate GT function
             GT_temp=eval_gt_func(GT_funcs{m},CM_thresh_temp);
             
             % reshape and put into large GT_data matrix (not that this
             % assumes that the output of the GT function does not have more
             % than 2 dimensions
             
             if ndims(GT_temp)>3
                 warning(['GT function ''' GT_funcs{m} ''' gives an output with more than 3-dimensions. This script assumes a max of 3 output dimensions. Output GT metrics will therfore not be properly indexed!']);
                 
             end
             GT_temp=permute(GT_temp,[4 5 1 2 3]);
             
             if ndims(GT_temp)==2
             GT_data{m}(s,th)=GT_temp;
                          
             elseif ndims(GT_temp)==3
             GT_data{m}(s,th,:)=GT_temp;
                          
             elseif ndims(GT_temp)==4
             GT_data{m}(s,th,:,:)=GT_temp;
             end
             
             
             
             if config.keep_CMthresh
                 config.CM_thresh(:,:,s,th)=CM_thresh_temp;
             end
             
         end
         
     end
     
end

config.thresh_abs=thresh_abs;
config.GT_funcs=GT_funcs;
            
            


%

% 


function [GT] = eval_gt_func(gt_func,CM)
try
    eval(gt_func);
catch exception
    exception
    error('GT functiion failed to execute with error (see above)');
end
GT=squeeze(GT);
if size(GT,1)==1 && ndims(GT)==2
    GT=GT';
end

function [CM_thresh,weights] = density_threshold_graph(CM,thresholds_density)

% thresholds a graph across a range of densities
%
% CM nxn matrix defing the graph and the edgeweights  derive the intial thresholds
% thresholds a 1xd vector of densities to threshold on. 
% 
% CM_thresh is a nxnxd binary array , representing the included regions at
% each threshold. 
% weights is a 1xd vector a values from the original connectivity matrix
% used to threshold the graphs. 
% 
% Note this function assumes conectivity matrix is symmetrical and that all
% weights have positive values. 




n_verts=length(CM);
n_thresh=length(thresholds_density);

% assume all NaNs should be 0
CM(find(isnan(CM)))=0;

keep_idx=find(triu(ones(n_verts),1));

% sort the CM values in ascending order
[CM_sorted CM_sort_idx]=sort(CM(keep_idx));
n_sorted=length(CM_sorted);


[CM_sort_i CM_sort_j]=ind2sub([n_verts n_verts],CM_sort_idx);

% compute the density of the untrhesholded matrix
dens0=nnz(CM(keep_idx))./length(keep_idx);

% estimate desnity for each point on CM_sorted
dens=linspace(1,0,n_sorted); 

% for each density threshold...

CM_thresh=false(n_verts,n_verts,n_thresh);

for k=1:n_thresh
    
    % check if current density threshold exceeds the original network
    % density
    
    if thresholds_density(k)>dens0
        weights(k)=0;
        CM_thresh(:,:,k)=CM>0;
        continue;
    end
    
    CM_thresh_temp=false(n_verts,n_verts);
    
    % find the nearest value in dens to the current density threshold
    dens_idx_temp=findnearest(thresholds_density(k),dens);
    
    % identify the corresponding CM weight
    weights(k)=CM_sorted(dens_idx_temp);

    % Set all value in CM_thresh at the current threhsodl or above to 1
    CM_thresh_temp(keep_idx(CM_sort_idx(dens_idx_temp:end)))=1;
    CM_thresh_temp=CM_thresh_temp+CM_thresh_temp';
    
    % concatent CM_thresh into a 3d binary array
    CM_thresh(:,:,k)=CM_thresh_temp;
    
    
    
end


    