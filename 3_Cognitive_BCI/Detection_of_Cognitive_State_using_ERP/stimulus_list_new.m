function   stimulus  = stimulus_list_new( mrk )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
cntStim1 = 1;
cntStim2 = 1;
cntStim3 = 1;
cntStim4 = 1;
cntStim5 = 1;
cntStim6 = 1;

for i = 1 : length(mrk.toe)
    switch mrk.toe(i)
        case 1
            stimulus.TargetBrake_stim(cntStim1) = mrk.pos(i);
            cntStim1 = cntStim1 + 1;
        case 2
            stimulus.NontargetBrake_stim(cntStim2) = mrk.pos(i);
            cntStim2 = cntStim2 + 1;
        case 3
            stimulus.NontargetLongBrake_stim(cntStim3) = mrk.pos(i);
            cntStim3 = cntStim3 + 1;
        case 4
            stimulus.Right_stim(cntStim4) = mrk.pos(i);
            cntStim4 = cntStim4 + 1;
        case 5
            stimulus.Left_stim(cntStim5) = mrk.pos(i);
            cntStim5 = cntStim5 + 1;
        case 9
            stimulus.Human_stim(cntStim6) = mrk.pos(i);
            cntStim6 = cntStim6 + 1;
    end
end

end

