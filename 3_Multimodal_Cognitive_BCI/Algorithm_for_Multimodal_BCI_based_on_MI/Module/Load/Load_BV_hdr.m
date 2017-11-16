function [ hdr ] = Load_BV_hdr( file )
%LOAD_BP_HDR Summary of this function goes here
%   Detailed explanation goes here
global BMI
[fid, message]=fopen(fullfile([file '.vhdr']));
if fid == -1
    [fid, message] = fopen(fullfile(pathname, lower(filename)));
    if fid == -1
        error(message)
    end;
end

hdr.DataFile= getEntry(fid, 'DataFile=', 0, [file '.eeg']);
getEntry(fid, '[Common Infos]');
hdr.DataFile= getEntry(fid, 'DataFile=', 0, [file '.eeg']);
hdr.MarkerFile= getEntry(fid, 'MarkerFile=', 0, [file '.vmrk']);
hdr.DataFormat= getEntry(fid, 'DataFormat=', 0);
hdr.DataOrientation= getEntry(fid, 'DataOrientation=', 0);
hdr.DataType= getEntry(fid, 'DataType=', 0);
getEntry(fid, '[Common Infos]');
hdr.NumberOfChannels= str2num(getEntry(fid, 'NumberOfChannels='));
hdr.DataPoints= getEntry(fid, 'DataPoints=', 0, '0');
getEntry(fid, '[Common Infos]');
hdr.SamplingInterval= str2num(getEntry(fid, 'SamplingInterval='));

getEntry(fid, '[Binary Infos]');
hdr.BinaryFormat= getEntry(fid, 'BinaryFormat=', 0);
hdr.UseBigEndianOrder= getEntry(fid, 'UseBigEndianOrder=', 0);

getEntry(fid, '[Channel Infos]');
hdr.chan=cell(1,hdr.NumberOfChannels);
for chan=1:hdr.NumberOfChannels
    temp=getEntry(fid,['Ch' num2str(chan) '='],0);
    chInfo=strsplit(temp,',');
    chInfo_=chInfo(~cellfun('isempty',chInfo));
    hdr.ChannelResolution(chan)=str2num(chInfo_{2});
    hdr.chan{chan}=chInfo_{1};
end

hdr.fs= 1000000/hdr.SamplingInterval;
%make a check if hdr.fs is a whole number
if(hdr.fs ~= round(hdr.fs))
  warning('file_readBVheader: hdr.fs was not a whole number: %f',hdr.fs); 
  hdr.fs= round(hdr.fs);
  hdr.SamplingInterval=1000000/hdr.fs;
end
fclose(fid);
end

function [entry, str]= getEntry(fid, keyword, mandatory, default_value)

if ~exist('mandatory','var'), mandatory=1; end
if ~exist('default_value','var'), default_value=[]; end
entry= 1;

if keyword(1)=='[',
    fseek(fid, 0, 'bof');
end
ok= 0;
while ~ok && ~feof(fid),
    str= fgets(fid);
    ok= strncmp(keyword, str, length(keyword));
end
if ~ok,
    if mandatory,
        error(sprintf('keyword <%s> not found', keyword));
    else
        entry= default_value;
        return;
    end
end
if keyword(end)=='=',
    entry= deblank(str(length(keyword)+1:end));
end


end
