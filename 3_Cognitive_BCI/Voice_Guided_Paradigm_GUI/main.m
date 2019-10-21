clc
clear
clear sound
%%%%%%%%%%%%%
session = 0;
%%%%%%%%%%%%%
[examples, questOrder, questCount, answers, isButtonActive, tcpClient] = init_experiment(session);
disp('실험을 시작해 주세요');
GUI1
