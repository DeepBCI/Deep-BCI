%% Information

clear all; 
clc;
sca;

%%%%% Please fill in before start %%%%%%

Subject='Ex_JH_HYS_170914_after';
trial=10;   block=12; % 3의 배수
Trigger=0;  Arduino=0;  Save=1;  ShowTable=1; Feedback=0;

durFirst=2; % 2
durCue=2; % 2
durTarget=3; % 3
durFeedback=2; % 2
durRest=2; % 2
durBlockRest=40; % 40
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(['Subject name:',Subject,'\n Block = ',num2str(block),'\n Trial = ',num2str(trial),...
    '\n Time per trial = ',num2str(durCue+1+durRest),' s ','\n Time per Block = '...
    ,num2str(durFirst+(durCue+1+durRest)*trial+durBlockRest),' s \n Total time = ',...
    num2str(fix((durFirst+(durCue+1+durRest)*trial+durBlockRest)*block/60)),' 분 ',...
    num2str(rem((durFirst+(durCue+1+durRest)*trial+durBlockRest)*block,60)),' 초 '])


total_start=GetSecs;

[Number,Cue,Target,First]=MakeNumberSet(trial,block,[0 1 2]);
%% Experiment

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% Get the screen numbers
screens = Screen('Screens');
% Screen('Preference', 'SkipSyncTests', 1);
% Select the external screen if it is present, else revert to the native
% screen
screenNumber = max(screens);

% Define black, white and grey
black = BlackIndex(screenNumber);
white = WhiteIndex(screenNumber);
grey = white / 2;

% Open an on screen window and color it grey
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, white);

% Measure the vertical refresh rate of the monitor
ifi = Screen('GetFlipInterval', window);

% Retreive the maximum priority number
topPriorityLevel = MaxPriority(window);

% Length of time and number of frames we will use for each drawing test
% numSecs = 1;
% numFrames = round(numSecs / ifi);

% Numer of frames to wait when specifying good timing
waitframes = 1;

% Set the blend funciton for the screen
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Get the size of the on screen window in pixels
% For help see: Screen WindowSize?
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Get the centre coordinate of the window in pixels
% For help see: help RectCenter
[xCenter, yCenter] = RectCenter(windowRect);



%%%%%%% Timing Information %%%%%%%%%

Num_trig=0;
numFirst=round(durFirst/ ifi);
numCue=round(durCue/ ifi);
numTarget=round(durTarget/ ifi);
numFeedback=round(durFeedback/ifi);
numRest=round(durRest/ ifi);
numBlockRest=round(durBlockRest/ ifi);


%%%%%%% Keyboard information %%%%%%%%%%

if Arduino
%%% Arduino 인식 준비 %%%
if ~isempty(instrfind)
    fclose(instrfind);
    delete(instrfind);
end
delete(instrfind)
a=arduino();
%%% Arduino %%%
end

% Define the keyboard keys that are listened for. We will be using the left
% and right arrow keys as response keys for the task and the escape key as
% a exit/reset key
escapeKey = KbName('ESCAPE');
left=KbName('LeftArrow');
right=KbName('RightArrow');
down=KbName('DownArrow');


%%%%%%% Experimental loop %%%%%%%%
% 결과 데이터 매트릭스. 1: accuracy, 2: time
respMat = nan(block,trial,2);
tstart=0;
total_word=0;
onset=zeros(1,1);
vbl=Screen('Flip',window);    
for i_item=1:block
        Screen('TextSize', window, 70);
        Screen('TextFont', window, 'Ariel');
        DrawFormattedText(window, double('시작하려면 아무 키나 눌러주세요'), 'center', 'center', black);
        Screen('Flip', window);
        KbStrokeWait;
        tstart=GetSecs;
for i=1:trial
    
    instruct={'0-back','1-back','2-back'};
    cue=Cue(i_item,i);
    target=Target(i_item,i,:);
    
    if i==1    
           vbl=Screen('Flip',window);     
for frame=1:numFirst
        Screen('TextSize', window, 50);
        Screen('TextFont', window, 'Ariel');
        DrawFormattedText(window,double(instruct{Number(i_item,trial,1)+1}),'center',screenYpixels*0.2,[1 0 1]);
        Screen('TextSize', window, 60);
        DrawFormattedText(window,num2str(First(i_item)),'center','center',[0 0 1]);
        vbl=Screen('Flip',window,vbl+(waitframes-0.5)*ifi);
end   
    end

vbl=Screen('Flip',window);    
for frame=1:numCue
        Screen('TextFont', window, 'Ariel');
        Screen('TextSize', window, 60);
        DrawFormattedText(window,num2str(cue),...
        'center', 'center', [0 0 0])
        vbl=Screen('Flip',window,vbl+(waitframes-0.5)*ifi);
end

    Beeper('high',0.5,0.3);
    respToBeMade =true;
    button_press1=[1,1,1,1]; button_press2=[1,1,1,1];  button_press3=[1,1,1,1];
    tStart = GetSecs;
    
    if Trigger
        Num_trig=Num_trig+1;
        Start_trig(Num_trig)=GetSecs;
    SP=serial('Com13','BaudRate',115200);
    fopen(SP);
    fprintf(SP,['mh',1010]);
    fclose(SP);
    delete(SP);
        Ent_trig(Num_trig)=GetSecs;
    else
        Num_trig=Num_trig+1;
        Start_trig(Num_trig)=GetSecs;
        Ent_trig(Num_trig)=GetSecs;

    end
    
    while respToBeMade==true
        % Draw the word
        DrawFormattedText(window,num2str(target(1)),screenXpixels*0.24,'center',[0 0 0]);    
        DrawFormattedText(window,num2str(target(2)),'center','center',[0 0 0]);    
        DrawFormattedText(window,num2str(target(3)),screenXpixels*0.7,'center',[0 0 0]);    
if Arduino
%%% %% Arduino button press %%%
            a8=readVoltage(a,'D8');
            a9=readVoltage(a,'D9');
            a10=readVoltage(a,'D10');
            
        button_press1=[button_press1(2:end),a8];  
        button_press2=[button_press2(2:end),a9];
        button_press3=[button_press3(2:end),a10];

        % Check the keyboard. The person should press the
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(escapeKey)
            ShowCursor;
            sca;
            return
        elseif button_press1==[0 0 0 0]
            response=1;
            respToBeMade=false;
        elseif button_press2==[0 0 0 0]
            response=2;
            respToBeMade=false;
        elseif button_press3==[0 0 0 0]
            response=3;
            respToBeMade=false;
        end   
else
       % Check the keyboard. The person should press the
        [keyIsDown,secs, keyCode] = KbCheck;
        if keyCode(escapeKey)
            ShowCursor;
            sca;
            return
        elseif keyCode(left)
            response=1;
            respToBeMade=false;
        elseif keyCode(down)
            response=2;
            respToBeMade=false;
        elseif keyCode(right)
            response=3;
            respToBeMade=false;
        end   
    end
    
        %%% Arduino button press 끝. 버튼 누르면 다음으로 넘어가는 기능 + 각 버튼 press가
        %%% response라는 변수에 저장됨 (1,2,3)
        % Flip to the screen
        vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);
    end
    tEnd = GetSecs;
    rt = tEnd - tStart;
    % Record the trial data into out data matrix
    respMat(i_item,i,1) = response;
    respMat(i_item,i,2) = rt;
    
    
    % Feedback
    if Feedback
vbl=Screen('Flip',window);        
for frame=1:numFeedback
    % Setup the text type for the window
LeftColor=zeros(1,3);   CenterColor=zeros(1,3); RightColor=zeros(1,3);
ColorMat=zeros(3,3);
ColorMat(Number(i_item,i,2),3)=1;

        DrawFormattedText(window,num2str(target(1)),screenXpixels*0.24,'center',ColorMat(1,:));    
        DrawFormattedText(window,num2str(target(2)),'center','center',ColorMat(2,:));    
        DrawFormattedText(window,num2str(target(3)),screenXpixels*0.7,'center',ColorMat(3,:));    
        vbl=Screen('Flip',window,vbl+(waitframes-0.5)*ifi);
        
end
    end

vbl=Screen('Flip',window);        
for frame=1:numRest
    % Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 36);
% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);
% Here we set the size of the arms of our fixation cross
fixCrossDimPix = 40;
% Now we set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor for us)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];
% Set the line width for our fixation cross
lineWidthPix = 4;  
Screen('DrawLines', window, allCoords,lineWidthPix, black, [xCenter yCenter], 2);
vbl=Screen('Flip',window,vbl+(waitframes-0.5)*ifi);
end
% Draw text in the upper portion of the screen with the default font in red
end


vbl=Screen('Flip',window);        
for frame=1:numBlockRest
    % Setup the text type for the window
Screen('TextFont', window, 'Ariel');
Screen('TextSize', window, 36);
% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);
% Here we set the size of the arms of our fixation cross
fixCrossDimPix = 40;
% Now we set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor for us)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];
% Set the line width for our fixation cross
lineWidthPix = 4;  
Screen('DrawLines', window, allCoords,lineWidthPix, black, [xCenter yCenter], 2);
vbl=Screen('Flip',window,vbl+(waitframes-0.5)*ifi);
end

Screen('Flip',window);
end 


if ShowTable
    
Tinstruct=Number(:,:,1);
Tanswer=Number(:,:,2);
Tresponse=respMat(:,:,1);
Trt=respMat(:,:,2);

Average_table=zeros(4,3); %Accuracy,rt_right,rt_wrong

for i=1:3
    
    Right_index=logical((Tinstruct==(i-1))&(Tanswer==Tresponse));
    Wrong_index=logical((Tinstruct==(i-1))&(Tanswer~=Tresponse));
    Average_table(1,i)=numel(find(Right_index))/numel(find(logical(Tinstruct==(i-1))));
    RT_total=Tresponse(Tinstruct==(i-1));    RT_right=Tresponse(Right_index);
    RT_wrong=Tresponse(Wrong_index);
    Average_table(2,i)=mean(RT_right);  Average_table(3,i)=mean(RT_wrong);
    Average_table(4,i)=mean(RT_total);
end
Accuracy=Average_table(1,:);
RT_right=Average_table(2,:);
RT_wrong=Average_table(3,:);
RT_total=Average_table(4,:);
Rows={'0-back','1-back','2-back'};
Cols={'Accuracy','RT_right','RT_wrong','RT_total'};
Table=table(Accuracy',RT_right',RT_wrong',RT_total','RowNames',Rows,'VariableNames',Cols)
end


close all;
sca;

total_end=GetSecs;
total_time=total_end-total_start;

if Save
save(Subject)
end



