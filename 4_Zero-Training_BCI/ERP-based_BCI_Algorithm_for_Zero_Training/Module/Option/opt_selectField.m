function [ out ] = opt_selectField( in, field )
% opt_selectField:
% 
% Description:
%     This function takes fields of input data structure altogether
% 
% Example:
%     field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
%     out=opt_selectField(EEG,field);
% 
% Input:
%     in    - EEG data structure
%     field - Fields you want to take
% Output:
%     out   - Data structure with fields combined
% 
% Min-Ho, Lee
% mhlee@image.korea.ac.kr
% 

if isempty(field)
    out=struct;
else
    for i=1:length(field)
        if isfield(in,field{i})
            out.(field{i})=in.(field{i});
        end
    end
    
end

end
