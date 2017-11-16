function [ dat ] = prep_filter( dat, varargin )
% prep_filter (Pre-processing procedure):
% 
% Description:
%     This function filters the data within specified frequency band
% 
% Example:
%    EEG.data=prep_filter(EEG.data, {'frequency', [7 13];'fs',100});
% 
% Input:
%     dat       - EEG data structure
% Option:
%     frequency - Frequency range that you want to filter
%     fs        - Sampling frequency
% Return:
%     dat       - Spectrally filtered data
% 
% Min-ho Lee
% mhlee@image.korea.ac.kr
%

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:}
end


switch isstruct(dat)
    case true %struct        
        if ~isfield(dat, 'x') && ~isfield(dat, 'fs')
            warning('Parameter is missing: dat.x or dat.fs');
        end
        
        if ~isfield(opt,'fs')
            if isfield(dat, 'fs')
                opt.fs=dat.fs;
            else
                error('Parameter is missing: fs');
            end
        end
        tDat=dat.x;
        if ndims(tDat)==3   %smt
            [nD nT nC]=size(tDat);
            tDat=reshape(tDat, [nD*nT,nC]);
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            tDat=reshape(tDat, [nD, nT,nC]);
            dat.x=tDat;
        elseif ndims(tDat)==2  %cnt
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            dat.x=tDat;
        end
        
        
    case false
        % add if dat is not struct
        tDat=dat;        
        if ndims(tDat)==3   %smt
            [nD nT nC]=size(tDat);
            tDat=reshape(tDat, [nD*nT,nC]);
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            tDat=reshape(tDat, [nD, nT,nC]);
            dat.x=tDat;
        elseif ndims(tDat)==2  %cnt
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            fld='x';
            dat=tDat;
        end
        
        % History
        if isfield(dat,'stack')
            c = mfilename('fullpath');
            c = strsplit(c,'\');
            dat.stack{end+1}=c{end};
        end
end



