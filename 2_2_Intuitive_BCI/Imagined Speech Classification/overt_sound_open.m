clear all
close all
clc;

%% overt_soundfile

cd('dir');
speech_file = dir('dir');
speech_file = struct2table(speech_file);
speech_filename = speech_file.name;

except_trial = ("Weather"|"is"|"it"|"How"|" "|"time"); % " ": except for sentence, space(blank)
delete_trial = strfind(speech_filename,except_trial);

for num_audio =1:length(speech_filename)
    if any(delete_trial{num_audio,1}) == 1
        speech_filename{num_audio,1} = [];
    end
end


speech_filename{1,1} = [];
speech_filename{2,1} = [];
speech_filename{end,1} = [];

O_n =1;
for ex_ovs = 1:length(speech_filename)
    if any(speech_filename{ex_ovs,1}) == 1
        overt_filename{O_n,1} = speech_filename{ex_ovs,1};
        O_n = O_n+1;
    end
end


wordlist = {'Hello','Yes','No', 'Help', 'Stop', 'Thank you'};
for i = 1:length(wordlist)
   overt_index = strfind(overt_filename,wordlist{1,i});
   w_array = cell2mat(overt_index);
   num_word(i,1) = length(w_array);
   overt_index = [];
end

for i = 1:length(overt_filename)
    tmp_overt_name{i,1} = 'xxx';
end

tmp_for_num_O = strcat(tmp_overt_name,overt_filename);

% task --> imagined, overt, whisper
for all_task = 1:1500
    num_starting{all_task,1} = ['xxx',num2str(all_task),'_']; % for classified name
end

list = 1;
for i = 1:length(num_starting)
    tmp_match = regexp(tmp_for_num_O,num_starting{i,1},'match');
    find_match = find(~cellfun(@isempty,tmp_match));
    if any(find_match) == 1
         overt_filelist{list,1} = overt_filename{find_match,1};
        list = list + 1;
    end
end


vec_audio = [];
speech_signal = [];
for ii = 1: length(overt_filelist)
    x = audioread(overt_filelist{ii,1});
    speech_signal(:,ii) = x;
    vec_audio = cat(1, vec_audio,x);
    clear x
end
Fs= 44100;
% Fs = 22050; % down-smapling




save(['D:\BTS_dataset\2022_10_11_BTS_denseNet_waveNet\KSW2_KJW8\2_word\sub8\day3\Overt_speech_data.mat'],['speech_signal'],['fs'],['overt_filelist'])

cd('D:\BTS_dataset\2022_10_11_BTS_denseNet_waveNet\KSW2_KJW8\2_word\sub3\day3')
filename = 'Overt_speech_audio_44100.wav';
audiowrite(filename,vec_audio,Fs);

