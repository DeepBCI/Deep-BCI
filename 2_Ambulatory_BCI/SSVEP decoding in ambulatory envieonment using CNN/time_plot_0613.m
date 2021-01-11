view_time = 140; %209
view_len = 5; %5
sub = 11; %12
ispeed = 4; % 4
% scale = 50;
% title_str = 'Raw';

% chan ={'L2','R7'};
% chan = 3:32;
% chan = {'F7','Fz','F8','FC5','FC6','C3','Cz','C4','CP5','CP6',...
%     'P3','Pz','P4','O1','Oz','O2'};
% chan = 3;
% chan = {'C3','Cz','C4'};

%% raw EEG cap
chan = {'C3','Cz','C4'};

scale = 50;
% ispeed = 4;
cnt = cap_cnt_filt{sub,ispeed};
% chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h1 = figure(1);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

% title(title_str)
h1.Position = [15 850 300 115];
% h1.Position = [15 700 300 200];

% h1.Position = [15 450 300 130];

%% isolated cap
scale = 2.5;
% ispeed = 2;
nCh = 2;
cnt = cap_ref_PCA{sub,ispeed};
cnt = cnt(1:nCh,view_time*100:(view_time+view_len)*100);

IMU_ = IMU_cnt{sub,ispeed};
IMU_ = proc_selectChannels(IMU_, 1:size(cnt,1));

h2 = figure(2);
plot_each_channel_bbci(IMU_, [0 0+view_len],...
    'data',cnt','en_text',false,'title',[],'scale',scale)
yticklabels(size(cnt,1):-1:1)
xticks(0:0+view_len);
xticklabels({0:view_len})
h2.Position = [15 650 300 100];


%% raw EEG ear
chan ={'L2','R7'};
scale = 100;
% ispeed = 4;
cnt = ear_cnt_filt{sub,ispeed};
% chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h3 = figure(3);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

% title(title_str)
h3.Position = [15 450 300 100];

% h1.Position = [15 450 300 130];

%% isolated ear
scale = .3;
% ispeed = 2;
nCh = 2;
cnt = ear_ref_PCA{sub,ispeed};
cnt = cnt(1:nCh,view_time*100:(view_time+view_len)*100);

IMU_ = IMU_cnt{sub,ispeed};
IMU_ = proc_selectChannels(IMU_, 1:size(cnt,1));

h4 = figure(4);
plot_each_channel_bbci(IMU_, [0 0+view_len],...
    'data',cnt','en_text',false,'title',[],'scale',scale)
yticklabels(size(cnt,1):-1:1)
xticks(0:0+view_len);
xticklabels({0:view_len})
h4.Position = [15 250 300 100];

%% IMU
scale = 20;
% ispeed = 4;
% chan = 1;
cnt = IMU_cnt{sub,ispeed};
cnt = proc_selectChannels(cnt, 1:2);
cnt.x = -cnt.x(:,1);

h5 = figure(5);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})
% h2.Position = [15 287 300 100];
h5.Position = [15 50 300 80];

%% proposed EEG
scale = 50;
% ispeed = 4;
cnt = cap_cnt_prop{sub,ispeed}; %i_cICA_PCA_AF
chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h6 = figure(6);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})
% title(title_str)
% h4.Position = [320 450 300 130];
h6.Position = [500 850 300 115];

%% SOTA EEG
scale = 50;
% ispeed = 4;
cnt = cap_cnt_CCA{sub,ispeed}; %i_cICA_PCA_AF
chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h7 = figure(7);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})
% title(title_str)
% h4.Position = [320 450 300 130];
h7.Position = [1000 850 300 115];


%% proposed EEG ear
chan ={'L2','R7'};
scale = 100;
% ispeed = 4;
cnt = ear_cnt_prop{sub,ispeed};
% chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h8 = figure(8);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

% title(title_str)
h8.Position = [500 450 300 100];

%% SOTA EEG ear
chan ={'L2','R7'};
scale = 100;
% ispeed = 4;
cnt = ear_cnt_CCA{sub,ispeed};
% chan = {'C3','Cz','C4'};
cnt = proc_selectChannels(cnt, chan);

% cnt = cap_cnt{sub,ispeed};

h9 = figure(9);
plot_each_channel_bbci(cnt, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

% title(title_str)
h9.Position = [1000 450 300 100];
