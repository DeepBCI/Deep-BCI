%% 64 trials
srate = 1000;
pre_frame = [-1.5 -0.3];
post_frame = [0.3 1.5];
for nmethod=1:4
    names = {'matrix_post_tort.mat', 'matrix_post_ozkurt.mat', 'matrix_post_PLV.mat', 'matrix_post_canolty.mat'};
    names_surr = {'matrix_post_tort_surrogates.mat', 'matrix_post_ozkurt_surrogates.mat', 'matrix_post_PLV_surrogates.mat', 'matrix_post_canolty_surrogates.mat'};
    
    names2 = {'matrix_pre_tort.mat', 'matrix_pre_ozkurt.mat', 'matrix_pre_PLV.mat', 'matrix_pre_canolty.mat'};
    names2_surr = {'matrix_pre_tort_surrogates.mat', 'matrix_pre_ozkurt_surrogates.mat', 'matrix_pre_PLV_surrogates.mat', 'matrix_pre_canolty_surrogates.mat'};
    iter_method = {'tort', 'ozkurt', 'PLV', 'canolty'};
    tic;
    disp(['Doing ' iter_method{nmethod}]);
    for nsb=1:16
        fprintf('Doing sub-%02d..\n', nsb);
        fname = sprintf('./samples_comparison/sub-%02d/VE_V1.mat', nsb);
        load(fname);
     
        EEG_pre = [];
        EEG_post =[];
        data3D_pre = [];
        data3D_post = [];
        data = [];
        time = [];
        % ---- converting VE_V1 to kh-friendly data structure
        for nbwindow=1:length(VE_V1.trial)
            data(1, :, nbwindow) = VE_V1.trial{nbwindow};
            time(1, :, nbwindow) = VE_V1.time{nbwindow};
            current_time = time(1, :, nbwindow);
            
        % now current version clip window here and segment data in
        % function
%             id_pre_frame = [find(current_time==pre_frame(1)), find(current_time==pre_frame(2))];
%             id_post_frame = [find(current_time==post_frame(1)), find(current_time==post_frame(2))];
%             data3D_pre(1, :, nbwindow) = data(:, id_pre_frame(1):id_pre_frame(2), nbwindow);
%             data3D_post(1, :, nbwindow) = data(:, id_post_frame(1):id_post_frame(2), nbwindow);
        end
        EEG_pre.data = data;
        EEG_pre.time = time;
        EEG_pre.srate = 1000;
        
        EEG_post.data = data;
        EEG_post.time = time;
        EEG_post.srate = 1000;
        
        [kh_matrix_post, kh_matrix_post_surr, kh_matrix_post_permute] = ...
            kh_detect_PAC_comodulation(EEG_post, [7 13], [34 100], iter_method{nmethod}, post_frame);
        [kh_matrix_pre, kh_matrix_pre_surr, kh_matrix_pre_permute] = ...
            kh_detect_PAC_comodulation(EEG_pre, [7 13], [34 100], iter_method{nmethod}, pre_frame);
        
        fname_post = sprintf('./kh_subsets/sub-%02d/%s', nsb, names{nmethod});
        fname_post_s = sprintf('./kh_subsets/sub-%02d/%s', nsb, names_surr{nmethod});
        
        fname_pre = sprintf('./kh_subsets/sub-%02d/%s', nsb, names2{nmethod});
        fname_pre_s = sprintf('./kh_subsets/sub-%02d/%s', nsb, names2_surr{nmethod});
        
        save(fname_post, 'kh_matrix_post');
        save(fname_post_s, 'kh_matrix_post_surr');
        
        save(fname_pre, 'kh_matrix_pre');
        save(fname_pre_s, 'kh_matrix_pre_surr');
    end
    toc;
end
disp('done');