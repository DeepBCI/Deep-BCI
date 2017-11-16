function [ mrk_orig ] = Load_BP_mrk( file, hdr, opt )
%LOAD_BP_MRK Summary of this function goes here
%   Detailed explanation goes here
global BMI;
if isempty(opt.fs)
    opt.fs=hdr.fs; % original sampling rate
end
mrk_all=readbvconf([file '.vmrk']);

% Only consider the 'Stimulus'.
n_s=1;
for i=1:length(mrk_all.markerinfos)
    strmrk=strsplit(mrk_all.markerinfos{i},',');
    switch strmrk{1}
        case 'Stimulus'
            Num_Stimulus=strrep(strmrk{2},'S','');
            mrk_orig.y(n_s)=str2num(Num_Stimulus);
            mrk_orig.t(n_s)=str2num(strmrk{3});
            mrk_orig.class{n_s}='Stimulus';
            n_s=n_s+1;
        case 'Comment'
            Num_Stimulus=strmrk{2};
            temp=strsplit(Num_Stimulus, ' ');
            if ~strcmp(temp{1},'actiCAP') && strcmp(temp{1},'no')
                disp(sprintf('no USB Connection to actiCAP, mrk %.0d',i));
            elseif ~strcmp(temp{1},'actiCAP') && ~strcmp(temp{1},'no')
                mrk_orig.y(n_s)=str2num(Num_Stimulus);
                mrk_orig.t(n_s)=str2num(strmrk{3});
                mrk_orig.class{n_s}='Stimulus';
                n_s=n_s+1;
            end           
    end
end

% marker resampling
lag = hdr.fs/opt.fs;
if lag~=round(lag) || lag<1,
    error('fs must be a positive integer divisor of every file''s fs');
end
mrk_orig.t= mrk_orig.t./lag;

end

