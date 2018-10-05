function  nonTargetSegment  = nontargetSegmentation_car_new(nonstimulus,normal,cnt, mrk ,ival)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ival_f = -cnt.fs*(ival(1,1)/1000);
ival_r = cnt.fs*(ival(1,2)/1000);

for m = 1: length(nonstimulus.start_stim)
    for k = 1 : size(cnt.x,2)
        cntStim = 1;
        cntStim1 = 1;
        cntRes = 1;
        cntNon = 1;
        offset = 200;
        stPoint = 1+nonstimulus.start_stim(m);
%         endPoint = 301+nonstimulus.start_stim(m);
        endPoint = ival_f+ival_r+1+nonstimulus.start_stim(m);
%         endPoint = 601+nonstimulus.start_stim(m);
        while 1
            if cntStim <= length(mrk.pos)
                if(endPoint > mrk.pos(cntStim) - 600 )
                    stPoint = mrk.pos(cntStim) + 601;
%                     endPoint = stPoint + 300;
                    endPoint = stPoint + ival_f + ival_r;
%                     endPoint = stPoint + 600;
                    cntStim = cntStim + 1;
                end
            end
            
              if cntRes <= length(normal)
                if(endPoint > normal(cntRes) - 600 )
                    stPoint = normal(cntRes) + 601;
%                     endPoint = stPoint + 300;
                    endPoint = stPoint + ival_f + ival_r;
%                     endPoint = stPoint + 600;
                    cntRes = cntRes + 1;
                end
              end
            
            if cntStim1 <= length(mrk.misc.pos)
                if mrk.misc.toe(cntStim1) == 6
                    cntStim1 = cntStim1 + 1;
                end
                if(endPoint > mrk.misc.pos(cntStim1) - 600)
                    stPoint = mrk.misc.pos(cntStim1) + 601;
                    %                      endPoint = stPoint + 300;
                    endPoint = stPoint + ival_f + ival_r;
                    %                     endPoint = stPoint + 600;
                    cntStim1 = cntStim1 + 1;
                end
            end
            if(endPoint > nonstimulus.end_stim(m))
                break;
            end
            nonTargetSegment(:, cntNon, k) = cnt.x(stPoint:endPoint,k);
            cntNon = cntNon + 1;
            stPoint = stPoint + offset;
            endPoint = endPoint + offset;
        end
        
    end
end

% nonTargetSegment.nonstim = nontarget;
% nonTarget = permute(nonTarget,[1 3 2]);
% %% Baseline Correction
% for i=1:size(cnt.x,2)
%     for n=1:length(nonTarget(1,:,1))
%         Mean = mean(nonTarget(1:20, n, i),1);
%         base_Nontarget(:, n, i) = nonTarget(:, n, i) - Mean;
%     end
% end


%% resegmentation
% base_Nontarget = base_Nontarget(121:160,:,:);

end

