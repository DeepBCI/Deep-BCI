% -------------------------------------------------------
% Update: 2020. 01. 08
% Soo-In Choi
% Detail: Multi-CSP, RLDAshrink, Online Classification, feedback
% -------------------------------------------------------


function [ out ] = Classification_online( eeg, online )
% ONLINE CLASSIFICATION
if ~isfield(online,'train'); warning('training module is not exist'); end
if ~isfield(online,'apply'); warning('applying module is not exist'); end

if ~isfield(online,'option'); disp('applying module is not exist');
    opt={
        'device', 'BrainVision'
        'paradigm', 'Cognitive'
        'Feedback','on'
        'host', 'USER-PC'
        'port','51244'
        }; % Host & Port number check! (연구실 장비 port_num 51244고정)
    opt=opt_cellToStruct(online.option{:});
else
    opt=opt_cellToStruct(online.option(:,:));
end

in=eeg;
if isfield(online,'train')
    for k=1:length(online.train)
        myFun=str2func(online.train{k});
        switch online.train{k}
            case 'proc_multiBandSpatialFilter' % Parameter check
                [out, CSP_W] = feval(myFun, in, online.train{k,2});
            case 'proc_variance' % epoch or continuous data structure
                out=feval(myFun, in);
            case 'proc_logarithm'
                out=feval(myFun, in);
            case 'proc_cspAuto'
                [out CSP_W, CSP_D] = feval(myFun, in, online.train{k,2},0);
            case 'func_train'
                % data (time x ch x trial) proc_variance => fv.x matrix (1 x ch x trail)
                % RLDAshrink func 'x' = fv.x (training matrix; ch x trail)
                %                 xsz = size(in.x);
                %                 fvsz = [prod(xsz(1:end-1)) xsz(end)];
                %                 in.x = reshape(in.x);
                C=feval(myFun, in, online.train{k,2});
            otherwise
                out=feval(myFun, in, online.train{k,2},0);
        end
        in=out;
    end
end

switch lower(opt.device)
    case 'brainvision'
        H = bv_open(opt.host);
        
end

while (true)
    dat=bv_read(H);
    % ---------------- Data acquisition (bbci) ----------------------------
    clab = H.clab;
    
    C= struct('b', 0);
    C.w= randn(length(clab), 1);
    bbci= struct;
    bbci.source.acquire_fcn= @bbci_acquire_bv;
    bbci.source.acquire_param= {'clab',clab, 'realtime', 2};
    bbci.feature.proc= {@proc_variance, @proc_logarithm};
    bbci.feature.ival= [-500 0];
    bbci.classifier.C= C;
    bbci.quit_condition.marker= 255;
    [source, marker] = bbci_apply_acquireData(source, bbci_source, marker)
    
    in=dat';
    
    if ~isempty(dat)
        if isfield(online,'apply')
            for k=1:length(online.apply)
                myFun=str2func(online.apply{k});
                switch online.apply{k}
                    case 'online_filterbank'
                        cnt.x = in;
                        cnt.fs = H.samplingInterval;
                        cnt.clab = H.channelNames;
                        state = [];
                        bands = [1 3; 4 7; 8 13; 14 29; 30 50];
                        [b, a] = butters(3, bands/(cnt.fs*2));
                        [in, state] = online_filterbank(cnt, state, b, a); % state: filter state
                    case 'proc_multiBandLinearDerivation'
                        out=feval(myFun, in, CSP_W);
                    case 'proc_variance' % epoch or continuous data structure
                        out=feval(myFun, in);
                    case 'proc_logarithm'
                        out=feval(myFun, in);
                    case 'apply_separatingHyperplane'
                        xsz = size(in.x);
                        out=feval(myFun, C.cf_param, reshape(in.x, [prod(xsz(1:end-1)) xsz(end)])); % input: (C y)
                    case 'loss_0_1'
                        loss_temp=feval(myFun, cnt.y, in); % loss of each trail
                        loss = mean(loss_temp);
                    otherwise
                        out=feval(myFun, in, online.apply{k,2});
                end
                in=out;
            end
            acc = round(1000*(1-loss))/10;
            out = acc; % trial classification loss 출력
            if strcmp(opt.Feedback,'on')
                % Feedback option
                as = wavread('beep_sound2.wav');
                ma = wavread('arithmetic.wav');
                ms = wavread('singing.wav');
                wa = wavread('word.wav');
                trial_num = 15; % 15 trials = 2 classes
                idx = randperm(length(trial_num));
                % ---------------------------------------------------------
                if cnt.y == out % trial label structure 확인 요
                    wavplay(ma); % 분류 결과 MA
                    bbci_trigger(10);
%                     loadpict([results1 '.jpg'],1);
%                     drawpict(1);
                    tic; pause(5); toc; % 5 s
                elseif cnt.y == out
                    wavplay(ms); % 분류 결과 MS
                    bbci_trigger(20);
%                     loadpict([results2 '.jpg'],1);
%                     drawpict(2);
                    tic; pause(5); toc;
                elseif cnt.y == out % 분류 결과 WA
                    wavplay(wa);
                    bbci_trigger(30);
%                     loadpict([results3 '.jpg'],1);
%                     drawpict(3);
                    tic; pause(5); toc;
                end
            end
        end
    end
end

end

