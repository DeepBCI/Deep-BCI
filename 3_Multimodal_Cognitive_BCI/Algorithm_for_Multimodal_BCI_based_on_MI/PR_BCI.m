function [ ] = PR_BCI( varargin )
%PR_BCI2 Summary of this function goes here
%   Detailed explanation goes here
if isempty(varargin) %find a PR_BCI directory when use this function in the inner directory
    FILE=[];
    CUR_FILE=strsplit(pwd,'\');
    for i=length(CUR_FILE):-1:1
        if  strfind(lower(CUR_FILE{i}),'PR_BCI');
            FILE=[];
            for j=1:i
                if j~=i
                    temp=strcat(CUR_FILE{j},'\');
                    FILE=strcat(FILE,temp);
                else
                    temp=CUR_FILE{j};
                    FILE=strcat(FILE,temp);
                end
                
            end
            break;
        else
            CUR_FILE{i}=[];
        end
    end
else
    FILE=[];
    FILE=varargin{1};
end
global BCI;
BCI.DIR=FILE;
BCI.EEG_DATA=[BCI.DIR '\BCI_data\DATA'];
BCI.CODE_DIR=[BCI.DIR '\BCI_modules'];
BCI.PARADIGM_DIR=[BCI.CODE_DIR '\Paradigms'] ;
BCI.IO_ADDR=hex2dec('C010');
BCI.IO_LIB=[BCI.CODE_DIR '\Options\Parallel_Port'];
% config_io;

if ischar(BCI.DIR)
addpath(genpath(BCI.DIR));
end
% cd(BCI.DIR);
end

