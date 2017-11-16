function [ cnt ] = opt_eegStruct( dat, field )
% opt_eegStruct:
% 
% Description:
%     This function converts the eeg data into a struct in OpenBMI format.
% 
% Example:
%    marker={'1','right';'2','left';'3','foot'};
%    field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
%    [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker',marker;'fs',fs});
%    CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
% 
% Input:
%     dat   - EEG data structure in a cell type
%     field - Selected necessary fields you want to get in a cell
% 
% Output:
%     cnt   - Raw EEG data in OpenBMI format with fields of input 'field'
% 
% Min-ho Lee
% mhlee@image.korea.ac.kr
%

if isempty(field)
    error('OpenBMI: There''s no field information')
elseif ~prod(ismember(field,{'x','t','fs','y_dec','y_logic','y_class','class', 'chan'}))
    error('OpenBMI: Unacceptable field name')
end

for i=1:length(field)
    [cnt.(field{i})]=[];
end

stc=struct;
if iscell(dat)
    nC=length(dat);
    for i=1:nC
        t_stc=opt_selectField(dat{i},field);
        stc=opt_catStruct(stc,t_stc);
    end
end

for i=1:length(field)
    if ~isempty(stc.(field{i}))
        cnt(:).(field{i})= stc.(field{i});     
    else
        warn_str=sprintf('OpenBMI: The field "%s" is empty', field{i});
        disp(warn_str);
    end
end

end

