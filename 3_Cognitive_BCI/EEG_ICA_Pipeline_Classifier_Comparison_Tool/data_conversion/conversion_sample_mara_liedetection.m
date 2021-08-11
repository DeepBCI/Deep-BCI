%%
%{


File description
This file is a sample file for preprocessing an eeg dataset to an ICA noise
corrected format that will be subsequently compatible to the ICA
classification training pipeline.
As you can see, the preprocessing scripting part of the pipeline is still
not fully streamlined; it is the classifcation & result collation part
downstream (mainly executed in python, as can be seen in /artifact_class/ )
that is currently streamlined to make it easy to compare classification
from multiple configurations of the same source dataset.

To rune this code you need the lie detection dataset matlab file, as well
as a Modified version of the MARA library (Winkler et al., 2011) that is
capable of processing segmented (non-continuous) EEG datafiles. You can
perform the modification on the library code yourself.


%}





% basic prep script: clear stuff and etc
close all;
clear;
startup;
addpath(['.\Mara\']);

root_dir = [EEG_MAT_DIR 'cogsys_transfer_lie\'];
label_root_dir = [root_dir 'train_ready_noica\'];

% original dataset to import epoch files
lie_path = [root_dir 'stim_1033\'];
subject_list_file = ['./session_list_lie_player.txt'];
nu_subject_list_file = ['./session_list_neuroscan.txt'];
% read bv seshlist
liesesh = textread(subject_list_file, '%s');


% load component files for task
ptcfiles = dir([root_dir 'components\*.mat']);
ptcfiles = {ptcfiles.name};
ptcfiles = liesesh;
ptc_size =  length(ptcfiles);

% also needed: labels, and base noica data to apply ica filter to
% thankfully it turns out that base noica data can be used to reject components after mara run 
labelfiles = dir([label_root_dir '\sbj_*.mat']);
labelfiles = {labelfiles.name};
%sorting filenames, because no leading zeros causes confusion in file
%sorting
[~, reindex] = sort( str2double( regexp( labelfiles, '\d+', 'match', 'once' )))
labelfiles = labelfiles(reindex) ;
labelfiles_size = length(labelfiles);

% for bart dataset clab info is not contained in datasetinfo.
% as such, there is no point in importing this file.
% as annoying as it is, we should load one file from the original dataset
clabfile = [root_dir 'train_ready_noica\datasetinfo.mat'];
load(clabfile);


for ptc=1:ptc_size
    
    % match with sbj name 
    %sbj_map{ptc} = ptcfiles{ptc}(1:end-4);
    sbj_map{ptc} = liesesh{ptc};
    load([root_dir 'components\' liesesh{ptc} '.mat']);
    
    % do wee need to stim out non stim trials here? Nope, that's already
    % done when creating component files from convert_color_ft_to_bbci.mat
    
    % since label is located within noica files (and that bbci format
    % requires the label within the datastructure), we need to load the
    % noica datafile for this person as well:
    
    
    % load noica train ready file
    load([label_root_dir labelfiles{ptc}]);
    % no need for epochs except the y variable, save some memory
    clear x_post x_pre
    

    
    
    % we still need to import abridged (x_pre+x_post) epochs from somewhere
    load([lie_path liesesh{ptc}]);
    
    

    
    % convert ft-format component data to bbci-format component data
    % which is simply an epo format but with components instead of channels
    cnt_cmp = convert_ft_comp_to_bbci(comp, 'component_file', {'stimInstLie, stimInstTruth, stimSponLie, stimSponTruth'}, label_4c_ind);
    
    cnt_cmp = proc_removeChannels(cnt_cmp, {'PO4', 'POz', 'Oz'});
    
    % proc mara : returns two variables : passed components and info
    % if you feed clab directly from comp it contains component names
    % instead of channels, and therefore causes error. Clab info must come
    % from some other variable: datasetinfo.mat contains clabs (load at
    % first few lines of the script to remove redundancy)
    [mara_goodcomp, mara_info] = proc_mara_nolas(cnt_cmp, data_clab, cnt_cmp.mix); % input is data, channels, and mixing array
    
    % based on selection we need to reject components, do it here:
    % no need to reconvert to ft for now? up to this point we kept the
    % original comp format. YUP.
    
    
    % seshepo is bbci, but ft_rejectcomponent requires ft format. better
    % convert them before merging
    %seshepo_ft = 
    
    
    % reforge data based on everything:
    cfg = [];
    cfg.component = mara_info.badcomp;
    
    data = ft_rejectcomponent(cfg, comp, data_w_eog);
    
    
    
    
    
    
    
    
    % save mara results somewher
    savepath = [root_dir 'mara_result\sbj_' num2str(ptc) '.mat'];
                
                [filepath, filename]= fileparts(savepath);
                if ~exist(filepath, 'dir')
                  [parentdir, newdir]=fileparts(filepath);
                  [status,msg]= mkdir(parentdir, newdir);
                  if status~=1
                    error(msg);
                  end
                end
    save(savepath, 'mara_goodcomp', 'mara_info');
    
    sbj_epo = convert_ft_to_bbci(data, ['lie/' sbj_map{ptc}], ...
        {'stimInstLie, stimInstTruth, stimSponLie, stimSponTruth'}, label_4c_ind);
    
    
    %% end of rejection: revert data format to ft and save
    % at the end of rejection(after mara) the components need to be
    % rejected, so convert them back to fieldtrip format (bbci may have
    % this feature but it is more convenient to do it in fieldtrip
    % considering preexisting code base of ours)
    
    % below should be done after rejection to match data formats between
    % comp and seshepo
    % remove P8 and Oz in order to equalized features space
    sbj_epo = proc_removeChannels(sbj_epo, {'PO4', 'POz', 'Oz'});

    

    % converting to trainable format
    e_pre = proc_selectIval(sbj_epo, [-500, 0]);
    e_post = proc_selectIval(sbj_epo, [0, 3000]);
    
    %x_pre = cat(3, x_pre, e_pre.x);
    %x_post = cat(3, x_post, e_post.x);
    x_pre = e_pre.x;
    x_post = e_post.x;
    %data_x = cat(3, data_x, sbj_epo.x);
  
    savepath = [root_dir 'train_ready_mara\sbj_' num2str(ptc) '.mat'];
                
                [filepath, filename]= fileparts(savepath);
                if ~exist(filepath, 'dir')
                  [parentdir, newdir]=fileparts(filepath);
                  [status,msg]= mkdir(parentdir, newdir);
                  if status~=1
                    error(msg);
                  end
                end
    save(savepath, '-v7.3', 'x_pre', 'x_post', 'label_2cis_ind', 'label_2clt_ind', 'label_4c_ind', 'label_inlt_ind', 'label_splt_ind');
    
    
    
end



