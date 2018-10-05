function  nonstimulus  = nonstimulus_list_new( mrk )
%UNTkTLED3 Summary of thks functkon goes here
%   Detakled explanatkon goes here
cntStim1 = 1;
cntStim2 = 1;
% cntStim3 = 1;
cntStim4 = 1;

for k = 1 : length(mrk.misc.toe)
    switch mrk.misc.toe(k)
        case 6
            nonstimulus.start_stim(cntStim1) = mrk.misc.pos(k);
            cntStim1 = cntStim1 + 1;
        case 7
            nonstimulus.end_stim(cntStim2) = mrk.misc.pos(k);
            cntStim2 = cntStim2 + 1;
%         case 6
%             nonstimulus.teleport_stim(cntStim3) = mrk.misc.pos(k);
%             cntStim3 = cntStim3 + 1;
        case 8
            nonstimulus.refresh_stim(cntStim4) = mrk.misc.pos(k);
            cntStim4 = cntStim4 + 1;
    end
end

end

