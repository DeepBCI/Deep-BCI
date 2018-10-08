clc; clear all; close all;

asmr = audioread('ASMR_waves_10min.mp3');
bb = audioread('BB_6.mp3');

%%
% bbCut = bb(1:length(asmr), :);
bbCutt = bb(1:8824500, :);
asmrCutt = asmr(1:8824500, :);

% ratio setting (1, 0.9, 0.8, ..., 0.1)
ratio = {0, -0.91, -1.94, -3.1, -4.44, -6, -7.95, -10.45, -14, -19.99};

asmr_level = 10^(0/20);
bb_level = 10^(ratio{8}/20); 

comb = (asmrCutt*asmr_level) + (bbCutt*bb_level);
combSound = audioplayer(comb, 44100, 16);

play(combSound)

