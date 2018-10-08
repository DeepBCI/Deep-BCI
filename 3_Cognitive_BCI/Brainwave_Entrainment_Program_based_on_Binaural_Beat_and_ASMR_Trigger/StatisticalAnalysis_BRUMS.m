clear all; close all; clc;

%% load data in excel

    %% before experiment
    BRUMS.before.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'B4:B12');
    BRUMS.before.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'C4:C12');
    BRUMS.before.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'D4:D12');
    BRUMS.before.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'E4:E12');
    BRUMS.before.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'F4:F12');
    BRUMS.before.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'G4:G12');
    BRUMS.before.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'H4:H12');
    BRUMS.before.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'I4:I12');

     %% Session 1 - Comb1
    BRUMS.S1_comb1.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'J4:J12');
    BRUMS.S1_comb1.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'K4:K12');
    BRUMS.S1_comb1.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'L4:L12');
    BRUMS.S1_comb1.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'M4:M12');
    BRUMS.S1_comb1.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'N4:N12');
    BRUMS.S1_comb1.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'O4:O12');
    BRUMS.S1_comb1.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'P4:P12');
    BRUMS.S1_comb1.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'Q4:Q12');

    %% Session 1 - Comb2
    BRUMS.S1_comb2.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'R4:R12');
    BRUMS.S1_comb2.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'S4:S12');
    BRUMS.S1_comb2.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'T4:T12');
    BRUMS.S1_comb2.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'U4:U12');
    BRUMS.S1_comb2.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'V4:V12');
    BRUMS.S1_comb2.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'W4:W12');
    BRUMS.S1_comb2.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'X4:X12');
    BRUMS.S1_comb2.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'Y4:Y12');

    %% Session 1 - Comb3
    BRUMS.S1_comb3.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'Z4:Z12');
    BRUMS.S1_comb3.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'AA4:AA12');
    BRUMS.S1_comb3.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'AB4:AB12');
    BRUMS.S1_comb3.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'AC4:AC12');
    BRUMS.S1_comb3.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'AD4:AD12');
    BRUMS.S1_comb3.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'AE4:AE12');
    BRUMS.S1_comb3.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'AF4:AF12');
    BRUMS.S1_comb3.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'AG4:AG12');
    
    %% Session 2 - BB
    BRUMS.S2_bb.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'AH4:AH12');
    BRUMS.S2_bb.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'AI4:AI12');
    BRUMS.S2_bb.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'AJ4:AJ12');
    BRUMS.S2_bb.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'AK4:AK12');
    BRUMS.S2_bb.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'AL4:AL12');
    BRUMS.S2_bb.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'AM4:AM12');
    BRUMS.S2_bb.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'AN4:AN12');
    BRUMS.S2_bb.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'AO4:AO12');

    %% Session 2 - ASMR
    BRUMS.S2_asmr.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'AP4:AP12');
    BRUMS.S2_asmr.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'AQ4:AQ12');
    BRUMS.S2_asmr.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'AR4:AR12');
    BRUMS.S2_asmr.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'AS4:AS12');
    BRUMS.S2_asmr.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'AT4:AT12');
    BRUMS.S2_asmr.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'AU4:AU12');
    BRUMS.S2_asmr.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'AV4:AV12');
    BRUMS.S2_asmr.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'AW4:AW12');

    %% Session 2 - Comb
    BRUMS.S2_comb.anger = xlsread('Questionnaire_Result_1.xlsx', 4, 'AX4:AX12');
    BRUMS.S2_comb.tension = xlsread('Questionnaire_Result_1.xlsx', 4, 'AY4:AY12');
    BRUMS.S2_comb.depression = xlsread('Questionnaire_Result_1.xlsx', 4, 'AZ4:AZ12');
    BRUMS.S2_comb.vigour = xlsread('Questionnaire_Result_1.xlsx', 4, 'BA4:BA12');
    BRUMS.S2_comb.fatigue = xlsread('Questionnaire_Result_1.xlsx', 4, 'BB4:BB12');
    BRUMS.S2_comb.confusion = xlsread('Questionnaire_Result_1.xlsx', 4, 'BC4:BC12');
    BRUMS.S2_comb.happy = xlsread('Questionnaire_Result_1.xlsx', 4, 'BD4:BD12');
    BRUMS.S2_comb.calmness = xlsread('Questionnaire_Result_1.xlsx', 4, 'BE4:BE12');
    
    
%% statistical analysis (t-test)
%% 8 Factors

    %% Session 1 - Comb1
    [Result.S1_comb1.anger, Pvalue.S1_comb1.anger] = ttest(BRUMS.S1_comb1.anger, BRUMS.before.anger);
    [Result.S1_comb1.tension, Pvalue.S1_comb1.tension] = ttest(BRUMS.S1_comb1.tension, BRUMS.before.tension);
    [Result.S1_comb1.depression, Pvalue.S1_comb1.depression] = ttest(BRUMS.S1_comb1.depression, BRUMS.before.depression);
    [Result.S1_comb1.vigour, Pvalue.S1_comb1.vigour] = ttest(BRUMS.S1_comb1.vigour, BRUMS.before.vigour);
    [Result.S1_comb1.fatigue, Pvalue.S1_comb1.fatigue] = ttest(BRUMS.S1_comb1.fatigue, BRUMS.before.fatigue);
    [Result.S1_comb1.confusion, Pvalue.S1_comb1.confusion] = ttest(BRUMS.S1_comb1.confusion, BRUMS.before.confusion);
    [Result.S1_comb1.happy, Pvalue.S1_comb1.happy] = ttest(BRUMS.S1_comb1.happy, BRUMS.before.happy);
    [Result.S1_comb1.calmness, Pvalue.S1_comb1.calmness] = ttest(BRUMS.S1_comb1.calmness, BRUMS.before.calmness);
   
    %% Session 1 - Comb2
    [Result.S1_comb2.anger, Pvalue.S1_comb2.anger] = ttest(BRUMS.S1_comb2.anger, BRUMS.before.anger);
    [Result.S1_comb2.tension, Pvalue.S1_comb2.tension] = ttest(BRUMS.S1_comb2.tension, BRUMS.before.tension);
    [Result.S1_comb2.depression, Pvalue.S1_comb2.depression] = ttest(BRUMS.S1_comb2.depression, BRUMS.before.depression);
    [Result.S1_comb2.vigour, Pvalue.S1_comb2.vigour] = ttest(BRUMS.S1_comb2.vigour, BRUMS.before.vigour);
    [Result.S1_comb2.fatigue, Pvalue.S1_comb2.fatigue] = ttest(BRUMS.S1_comb2.fatigue, BRUMS.before.fatigue);
    [Result.S1_comb2.confusion, Pvalue.S1_comb2.confusion] = ttest(BRUMS.S1_comb2.confusion, BRUMS.before.confusion);
    [Result.S1_comb2.happy, Pvalue.S1_comb2.happy] = ttest(BRUMS.S1_comb2.happy, BRUMS.before.happy);
    [Result.S1_comb2.calmness, Pvalue.S1_comb2.calmness] = ttest(BRUMS.S1_comb2.calmness, BRUMS.before.calmness);

    %% Session 1 - Comb3
    [Result.S1_comb3.anger, Pvalue.S1_comb3.anger] = ttest(BRUMS.S1_comb3.anger, BRUMS.before.anger);
    [Result.S1_comb3.tension, Pvalue.S1_comb3.tension] = ttest(BRUMS.S1_comb3.tension, BRUMS.before.tension);
    [Result.S1_comb3.depression, Pvalue.S1_comb3.depression] = ttest(BRUMS.S1_comb3.depression, BRUMS.before.depression);
    [Result.S1_comb3.vigour, Pvalue.S1_comb3.vigour] = ttest(BRUMS.S1_comb3.vigour, BRUMS.before.vigour);
    [Result.S1_comb3.fatigue, Pvalue.S1_comb3.fatigue] = ttest(BRUMS.S1_comb3.fatigue, BRUMS.before.fatigue);
    [Result.S1_comb3.confusion, Pvalue.S1_comb3.confusion] = ttest(BRUMS.S1_comb3.confusion, BRUMS.before.confusion);
    [Result.S1_comb3.happy, Pvalue.S1_comb3.happy] = ttest(BRUMS.S1_comb3.happy, BRUMS.before.happy);
    [Result.S1_comb3.calmness, Pvalue.S1_comb3.calmness] = ttest(BRUMS.S1_comb3.calmness, BRUMS.before.calmness);

    %% Session 2 - BB
    [Result.S2_bb.anger, Pvalue.S2_bb.anger] = ttest(BRUMS.S2_bb.anger, BRUMS.before.anger);
    [Result.S2_bb.tension, Pvalue.S2_bb.tension] = ttest(BRUMS.S2_bb.tension, BRUMS.before.tension);
    [Result.S2_bb.depression, Pvalue.S2_bb.depression] = ttest(BRUMS.S2_bb.depression, BRUMS.before.depression);
    [Result.S2_bb.vigour, Pvalue.S2_bb.vigour] = ttest(BRUMS.S2_bb.vigour, BRUMS.before.vigour);
    [Result.S2_bb.fatigue, Pvalue.S2_bb.fatigue] = ttest(BRUMS.S2_bb.fatigue, BRUMS.before.fatigue);
    [Result.S2_bb.confusion, Pvalue.S2_bb.confusion] = ttest(BRUMS.S2_bb.confusion, BRUMS.before.confusion);
    [Result.S2_bb.happy, Pvalue.S2_bb.happy] = ttest(BRUMS.S2_bb.happy, BRUMS.before.happy);
    [Result.S2_bb.calmness, Pvalue.S2_bb.calmness] = ttest(BRUMS.S2_bb.calmness, BRUMS.before.calmness);
    %% Session 2 - ASMR
    [Result.S2_asmr.anger, Pvalue.S2_asmr.anger] = ttest(BRUMS.S2_asmr.anger, BRUMS.before.anger);
    [Result.S2_asmr.tension, Pvalue.S2_asmr.tension] = ttest(BRUMS.S2_asmr.tension, BRUMS.before.tension);
    [Result.S2_asmr.depression, Pvalue.S2_asmr.depression] = ttest(BRUMS.S2_asmr.depression, BRUMS.before.depression);
    [Result.S2_asmr.vigour, Pvalue.S2_asmr.vigour] = ttest(BRUMS.S2_asmr.vigour, BRUMS.before.vigour);
    [Result.S2_asmr.fatigue, Pvalue.S2_asmr.fatigue] = ttest(BRUMS.S2_asmr.fatigue, BRUMS.before.fatigue);
    [Result.S2_asmr.confusion, Pvalue.S2_asmr.confusion] = ttest(BRUMS.S2_asmr.confusion, BRUMS.before.confusion);
    [Result.S2_asmr.happy, Pvalue.S2_asmr.happy] = ttest(BRUMS.S2_asmr.happy, BRUMS.before.happy);
    [Result.S2_asmr.calmness, Pvalue.S2_asmr.calmness] = ttest(BRUMS.S2_asmr.calmness, BRUMS.before.calmness);

    %% Session 2 - Comb
    [Result.S2_comb.anger, Pvalue.S2_comb.anger] = ttest(BRUMS.S2_comb.anger, BRUMS.before.anger);
    [Result.S2_comb.tension, Pvalue.S2_comb.tension] = ttest(BRUMS.S2_comb.tension, BRUMS.before.tension);
    [Result.S2_comb.depression, Pvalue.S2_comb.depression] = ttest(BRUMS.S2_comb.depression, BRUMS.before.depression);
    [Result.S2_comb.vigour, Pvalue.S2_comb.vigour] = ttest(BRUMS.S2_comb.vigour, BRUMS.before.vigour);
    [Result.S2_comb.fatigue, Pvalue.S2_comb.fatigue] = ttest(BRUMS.S2_comb.fatigue, BRUMS.before.fatigue);
    [Result.S2_comb.confusion, Pvalue.S2_comb.confusion] = ttest(BRUMS.S2_comb.confusion, BRUMS.before.confusion);
    [Result.S2_comb.happy, Pvalue.S2_comb.happy] = ttest(BRUMS.S2_comb.happy, BRUMS.before.happy);
    [Result.S2_comb.calmness, Pvalue.S2_comb.calmness] = ttest(BRUMS.S2_comb.calmness, BRUMS.before.calmness);

