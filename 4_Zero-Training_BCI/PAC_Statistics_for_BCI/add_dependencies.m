% add custom functino path
function_path = './functions';
custom_function_path = './made_functions';

% add opensource toolbox path
eeglab_path = ''; % your eeglab root directory
fieldtrip_path = ''; % your fieldtrip root directory

addpath(function_path);
addpath(custom_functino_path);
addpath(eeglab_path);
addpath(fieldtrip_path);

% add EEGLAB and FIELDTRIP subdirectories using calling
eeglab;
ft_defaults;