function [ eeg hdr] = Load_BV_data( file, hdr, opt)
%LOAD_BP_DATA Summary of this function goes here
%   Detailed explanation goes here

[fid, message] = fopen(fullfile([file '.eeg']));
if fid == -1
    [fid, message] = fopen(fullfile(pathname, lower(filename)));
    if fid == -1
        error(message)
    end
end;

if isempty(opt.fs)
    opt.fs=hdr.fs;
end

% Binary Infos %BCILAB, bps=bytespersample
if strcmpi(lower(hdr.DataFormat), 'binary')
    switch lower(hdr.BinaryFormat)
        case 'int_16',        binformat = 'int16'; bps = 2;
        case 'uint_16',       binformat = 'uint16'; bps = 2;
        case 'ieee_float_32', binformat = 'float32'; bps = 4;
        otherwise, error('Unsupported binary format');
    end
end

% EEG default setting
fseek(fid, 0, 'eof');
hdr.Datapoints = ftell(fid) / (hdr.NumberOfChannels * bps);
eeg=set_EEG_defualt(hdr,opt);

%EEG data load
switch lower(hdr.DataOrientation)
    case 'multiplexed'
        fseek(fid, 0, 'bof');
        eeg.x= fread(fid, [hdr.NumberOfChannels, hdr.Datapoints], [binformat '=>float32']);
    otherwise
        waring('Not implemented');
end
fclose(fid);

%re-scale
for chan=1:hdr.NumberOfChannels
    eeg.x(chan,:)=eeg.x(chan,:)*hdr.ChannelResolution(chan);
end

% Convert to EEG.data to double for MATLAB < R14
eeg.x = double(eeg.x);
%downsampling
lag=hdr.fs/opt.fs;
if lag~=round(lag) || lag<1,
    error('fs should be the integer');
end
for chan=1:hdr.NumberOfChannels
    resampled_eeg(chan,:)=chan_resampling(eeg.x(chan,:),opt.fs,hdr.fs);
end
if hdr.fs ~=opt.fs
    eeg.fs=opt.fs;
    eeg.orig_fs= hdr.fs;
    hdr.orig_fs=hdr.fs;
    hdr.fs=opt.fs;
end
% eeg.x=resampled_eeg;
% eeg.x=eeg.x'
eeg.x=resampled_eeg';
end

function [dat] = chan_resampling(dat,p,q)
dat=resample(dat,p,q);
end

