function [ eeg ] = set_EEG_defualt(hdr, opt)
% set_EEG_defualt:
% 
% Description:
%     This function sets basic frame of eeg data structure.
% 
% Example:
%     eeg=set_EEG_defualt(hdr,opt);
% 
% Input:
%     hdr - header of the data
%     opt - information of the data (device, marker, sampling frequency)
% Output:
%     eeg - eeg data structure, with basic frame
% 
% Min-ho Lee
% mhlee@image.korea.ac.kr
%


eeg.x= zeros(hdr.NumberOfChannels,100000);

if ~isfield(hdr,'NumberOfChannels') || ~isfield(opt,'fs') 
    disp('Important parameter is missing, check the hdr.NumberOfChannels and opt.fs')
end
eeg.nCh=hdr.NumberOfChannels;
eeg.chSet=[];
eeg.stack={};

end

