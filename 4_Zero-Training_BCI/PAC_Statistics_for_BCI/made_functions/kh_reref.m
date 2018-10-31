% kh_reref() - do re-refrencing EEG data to some other common
%                  reference or to average reference
%
% Usage: 
%   >> data_out = kh_reref(data_in, ref); 
%                   %convert EEG data according to ref
% Inputs:
%    data_in -  2-D data matrix (chans x frames)
%    ref - reference channel number(s)
%      1) 'common' - compute average reference
%      2) [X Y Z, ...] - re-reference to the average of channel X Y Z..
%
% Outputs:
%    data_out - input data converted to the new reference
%
% Author: Kyungho Won, BioComputing Lab, 2017.08.29

function data_out = kh_reref(data_in, ref, varargin)

if nargin<2
    help re_reference
    return
end

disp('System: EEG re-referencing ...');
if strcmp(ref, 'common')
    common_avg = mean(data_in, 1);
%     common_avg = repmat(common_avg, size(data_in, 1), 1);
    
    for ch=1:size(data_in, 1)
        data_out(ch, :) = (data_in(ch, :) - common_avg) - sum((data_in(ch, :) - common_avg)) / size(data_in, 1);
    end
    
%     data_out = (data_in - common_avg) - sum(data_in - common_avg) / size(data_in, 1);    
    disp('System: Common average reference applied');
else
    specific_avg = mean(data_in(ref, :), 1);
%     specific_avg = repmat(specific_avg, size(data_in, 1), 1);

    for ch=1:size(data_in, 1)
        data_out(ch, :) = (data_in(ch, :) - specific_avg) - sum((data_in(ch, :) - specific_avg)) / size(data_in, 1);
    end

%     data_out = (data_in - specific_avg) - sum((data_in - specific_avg)) / size(data_in,1); 
    disp(['System: ', num2str(ref), ' average reference applied']);
end

end