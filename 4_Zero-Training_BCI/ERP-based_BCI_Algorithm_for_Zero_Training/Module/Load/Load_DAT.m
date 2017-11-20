function [ dat, marker, hdr] = Load_EEG( file, varargin )
%LOAD_ Summary of this function goes here
%   Detailed explanation goes here

% opt=opt_proplistToStruct(varargin{:});
if ~isempty(varargin)
    opt=opt_cellToStruct(varargin{:});
else % set default parameters here
    opt.device='brainVision';
end

if ~isfield(opt,'fs')
    opt.fs=[];% seting original sampling rate from Load_BV_hdr
end

switch lower(opt.device)
    case 'brainvision'
        hdr=Load_BV_hdr(file);disp('Loading EEG header file..');
        marker=Load_BV_mrk(file, hdr, opt);disp('Loading Marker file..');
        [dat hdr]=Load_BV_data(file, hdr, opt);disp('Loading EEG data..');        
        if isfield(opt,'marker')
            [marker]=prep_defineClass(marker,opt.marker);
        end
    case 'emotive'
        
end
disp('Data loaded!');
end

