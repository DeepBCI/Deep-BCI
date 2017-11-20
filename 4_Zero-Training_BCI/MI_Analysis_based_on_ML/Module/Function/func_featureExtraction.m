function [ out ] = func_featureExtraction( dat, varargin )
%func_featureExtraction:
%    Feature extracting the epoched or continous data.
% Example:
%      out=func_featureExtraction(smt, {'feature','logvar'});
%  Input:
%      dat - Data structure of epoched
%     
% Options:
%     feature - 'logvar' or 'erpmean'

% Returs:
%    out  - Extracted features.

if iscell(varargin)
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:};
end

if isstruct(dat)
    tDat=dat.x;
else
    tDat=dat;
end

switch lower(opt.feature)
    case 'logvar'
        tDat=squeeze(log(var(tDat)));
        tDat=tDat';
        
    case 'erpmean'
        %         [nDat, nTrials, nChans]= size(tDat);
        %         T=param{2};
        %         nMeans= round(nDat/T);
        %         dat_=zeros(nMeans, nTrials, nChans);
        %         for i=1:nMeans
        %             if i==nMeans
        %                 temp=mean(tDat((i*T-T)+1:end,:,:),1);
        %             else
        %                 temp=mean(tDat((i*T-T)+1:i*T,:,:),1);
        %             end
        %             dat_(i,:,:)=temp; temp=[];
        %         end
        %         [nDat, nTrials, nChans]= size(dat_);
        %         tDat= reshape(permute(dat_,[1 3 2]), [nDat*nChans nTrials]);
        
        % smkim
        if isfield(opt,'nMeans')
            nM=opt.nMeans;
            tDat=prep_erpMeans(tDat,{'nMeans',nM});
        elseif isfield(opt,'nSamples')
            nS=opt.nSamples;
            tDat=prep_erpMeans(tDat,{'nSamples',nS});
        else
            warning('OpenBMI: Options for prep_erpMeans (e.g. nMeans or nSamples) are needed'),return
        end
end

if isstruct(dat)
    dat.x=tDat;
else
    dat=tDat;
end

if isfield(dat,'stack') %% put in the function
    % stack
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat.stack{end+1}=c{end};
end
out=dat;

end

