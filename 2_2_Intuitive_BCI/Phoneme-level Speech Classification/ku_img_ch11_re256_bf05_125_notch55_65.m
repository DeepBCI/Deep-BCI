dList = dir('D:\datasets\bts_data\sess1_Imagined_Speech\*.vhdr');
k = length(dList);

for i = 1:k
	EEG = pop_loadbv('D:\datasets\bts_data\sess1_Imagined_Speech\', dList(i).name);
	EEG = pop_select( EEG, 'channel',{'C4','FC3','FC1','F5','C3','F7','FT7','CZ','P3','T7','C5'});
	EEG = pop_resample( EEG, 256);
	EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',125,'plotfreqz',1);
	EEG = pop_eegfiltnew(EEG, 'locutoff',55,'hicutoff',65,'revfilt',1,'plotfreqz',1);
	EEG = pop_chanedit(EEG, 'lookup','D:\eeglab2021.1\plugins\dipfit\standard_BEM\elec\standard_1005.elc');
	EEG = pop_epoch( EEG, {  'S  1'  'S  2'  'S  3'  'S  4'  'S  5'  'S  6'  'S  7'  'S  8'  'S  9'  'S 10'  'S 11'  'S 12'  'S 13'  }, [0  2], 'newname', 'ch_resample_filter_notch_epochs', 'epochinfo', 'yes');
	EEG = pop_rmbase( EEG, [],[]);
	
	for j = 1:13
		EEG_sep = pop_selectevent( EEG, 'type',{sprintf('S%3s',num2str(j))},'deleteevents','off','deleteepochs','on','invertepochs','off');
        save([num2str((i), '%02d'), '_', num2str((j), '%02d')],'EEG_sep')
	end
end	