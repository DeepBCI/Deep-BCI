This project allows users to play mole cathcing game using their brain activity on single/two-player mode.

Please note that this game was modified and re-coded from the original opensource project for developing brain signal based game play;
URL: https://github.com/jineyne/Mole-Game

The external progrem version : 
OpenViBE - ver.2.1.0 , Unity - ver.2018.3.7f1 , Matlab - ver.2015b, 32bit

If your Maltab version is low, canoncorr function might not work. Then add "canoncorr.m" in path. 

[How to use]

1. open the file in OpenViBE_scenario at "openvibe-designer", OpenViBE version is 2.1.0 
2. modify Matlab working directory in Matlab scripting to Matlab_scripts, and matching name of function and script name.
3. run "Mole Game.exe"
4. run "openvibe-acquisition-server.cmd" ,and set the driver to LabStreamingLayer, then check signal stream and marker stream in driver properties 
5. start "openvibe-designer", then start in "Mole Game.exe"