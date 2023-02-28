dList = dir('D:\overt_words\*.vhdr');
k = length(dList);

for i = 5:k
	EEG = pop_loadbv('D:\overt_words\', dList(i).name);
	EEG = pop_select( EEG, 'nochannel',{'EOG_UP','EOG_DOWN','EOG_LEFT','EOG_RIGHT','EMG_L_C1','EMG_L_C2','EMG_L_C3','EMG_L_M','EMG_U_M','EMG_D_M','EMG_R_M','EMG_R_C1','EMG_R_C2','EMG_R_C3','EMG_REF','NEW_ADD'});
	%EEG = pop_resample( EEG, 256);
	EEG = pop_eegfiltnew(EEG, 'locutoff',30,'hicutoff',120,'plotfreqz',1);
	EEG = pop_eegfiltnew(EEG, 'locutoff',59.9,'hicutoff',60.1,'revfilt',1,'plotfreqz',1);
	EEG = pop_chanedit(EEG, 'lookup','D:\eeglab2021.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc');
	%EEG = pop_runica(EEG, 'icatype', 'runica', 'interrupt','on');
	%EEG = pop_iclabel(EEG, 'default');
	%EEG = pop_icflag(EEG, [NaN NaN;0.7 1;0.7 1;0.7 1;0.7 1;0.7 1;0.7 1; 0.7 1]);
	%EEG = pop_subcomp( EEG, [], 0);
	EEG = pop_epoch( EEG, {  'S  1'  'S  2'  'S  3'  'S  4'  'S  5'  'S  6'  'S  7'  'S  8'  'S  9'  'S 10'  'S 11'  'S 12'  'S 13'  }, [0  2], 'newname', 'ch_resample_filter_notch_epochs', 'epochinfo', 'yes');
	EEG = pop_rmbase( EEG, [],[]);

	for j = 1:13
		EEG_sep = pop_selectevent( EEG, 'type',{sprintf('S%3s',num2str(j))},'deleteevents','off','deleteepochs','on','invertepochs','off');
		save([num2str((i), '%02d'), '_', num2str((j), '%02d')],'EEG_sep')
	end
end	
