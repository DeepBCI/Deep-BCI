function varargout = MainGUI(varargin)
% MAINGUI MATLAB code for MainGUI.fig
%      MAINGUI, by itself, creates a new MAINGUI or raises the existing
%      singleton*.
%
%      H = MAINGUI returns the handle to a new MAINGUI or the handle to
%      the existing singleton*.
%
%      MAINGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAINGUI.M with the given input arguments.
%
%      MAINGUI('Property','Value',...) creates a new MAINGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MainGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MainGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MainGUI

% Last Modified by GUIDE v2.5 31-Oct-2017 21:12:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MainGUI_OpeningFcn, ...
    'gui_OutputFcn',  @MainGUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MainGUI is made visible.
function MainGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MainGUI (see VARARGIN)

% Choose default command line output for MainGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MainGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MainGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in btn1.
function btn1_Callback(hObject, eventdata, handles)
% hObject    handle to btn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
persistent state;               %1 이면 실행중, 0이면 꺼짐
persistent graph_h1; 
persistent graph_h2;
persistent doing;               % 반복문이 진행중인 동안은 1이 되는 변수
persistent breaknow;            % 반복문이 진행중이라면 반복문을 강제종료
global recording;

if isempty(breaknow)
    breaknow = 0;
end 

if isempty(doing)
    doing = 0;
end

if isempty(state)
    state = 0;              % Plotting 중이면 1, 그렇지 않으면 0
end

if state == 0
    
    %set(handles.btn1,'Stop','running','enable','off');
    state = 1;
    
    delete(instrfindall)           % 현재 연결되어 있는 시리얼포트들을 다 지웁니다.
    s2 = serial('COM3');          % 아두이노에서 연결했던 시리얼포트 번호를 입력합니다.
    s2.InputBufferSize = 1024;
    set(s2,'BaudRate',9600)
    fopen(s2);                     % 포트를 엽니다.
    axes(handles.axes1);
    
    pack = fscanf(s2);

    [trialnum, pow1num, pow2num] = getPow(pack); 
    
    Y1 = zeros(1,100);       %세로축
    Y2 = zeros(1,100);
    hbo = zeros(1,100);
    hb = zeros(1,100);
    X = 0 : 0.1 : 9.9;        %가로축
    
    Y1(1) = pow1num;
    Y2(1) = pow2num;
    
   
    
   % [hbo(1),hb(1)] = MBLL([0],[0],3,850,735);
    
    file_h=fopen('rawdata.txt','w');
    fprintf(file_h,'%d %d %d \n', trialnum, pow1num, pow2num);
    fclose(file_h);
 
        file_h = fopen('rawdata_1.txt','w');
        fprintf(file_h,'');
        fclose(file_h);
 
    
    startTime = clock;
   
    while (1)
       
        set(handles.text3, 'String', sprintf('%d',int32(etime(clock,startTime))));              %%현재시간 표시
        
        if (etime(clock,startTime) <= 20)
            set(handles.text2, 'String', sprintf('REST'));
             msgreset = false;
             
        elseif(etime(clock,startTime) <= 40)
            if msgreset == false
            rng('shuffle');
            set(handles.text2, 'String', sprintf('%d - %d',randi([100,999]), randi([3,9])));
            msgreset = true;    
            pause(0.01); 
            else pause(0.01);
            end
     
        elseif(etime(clock,startTime) <= 60)
            set(handles.text2, 'String', sprintf('REST'));
            if msgreset == true
                msgreset = false;
            end
            
        elseif(etime(clock,startTime) <= 80)
            if msgreset == false
            rng('shuffle');
            set(handles.text2, 'String', sprintf('%d - %d',randi([100,999]), randi([3,9])));
            msgreset = true;    
            pause(0.01); 
            else pause(0.01);
            end
            
        elseif(etime(clock,startTime) <= 100)
            set(handles.text2, 'String', sprintf('REST'));
            if msgreset == true
                msgreset = false;
            end
            
        elseif(etime(clock,startTime) <= 120)
            if msgreset == false
            rng('shuffle');
            set(handles.text2, 'String', sprintf('%d - %d',randi([100,999]), randi([3,9])));
            msgreset = true;    
            pause(0.01); 
            else pause(0.01);
            end
            
        elseif(etime(clock,startTime) <= 140)
            set(handles.text2, 'String', sprintf('REST'));
            if msgreset == true
                msgreset = false;
            end
                
        elseif(etime(clock,startTime) <= 160)
            if msgreset == false
            rng('shuffle');
            set(handles.text2, 'String', sprintf('%d - %d',randi([100,999]), randi([3,9])));
            msgreset = true;    
            pause(0.01); 
            else pause(0.01);
            end
            
        elseif(etime(clock,startTime) <= 180)
            set(handles.text2, 'String', sprintf('REST'));
            if msgreset == true
                msgreset = false;
            end
            
        elseif(etime(clock,startTime) <= 200)
            if msgreset == false
            rng('shuffle');
            set(handles.text2, 'String', sprintf('%d - %d',randi([100,999]), randi([3,9])));
            msgreset = true;    
            pause(0.01); 
            else pause(0.01);
            end
            
        elseif(etime(clock,startTime) <= 220)
            set(handles.text2, 'String', sprintf('END'));
            if msgreset == true
                msgreset = false;
            end
        
            
        end
        
        
        for n=1:99
            Y1(101-n) = Y1(100-n);
            Y2(101-n) = Y2(100-n);
            hb(101-n) = hb(100-n);
            hbo(101-n) = hbo(100-n);
        end
        
          pack = fscanf(s2);
        
         [trialnum, pow1num, pow2num] = getPow(pack); 

        Y1(1,1) = pow1num;                              %850wavelength
        Y2(1,1) = pow2num;                              %735wavelength
            
        axes(handles.axes1);
        graph_h1 = plot(X,Y1,'r',X, Y2,'b');
        grid on;
       % ylim([0, 1100]);
        xlabel('Time [ sec ]');
        ylabel('amplitude');
        title('Light');
        drawnow;
        
        file_h=fopen('rawdata.txt','a');
        fprintf(file_h,'%d %d %d \n', trialnum, pow1num, pow2num);
        fclose(file_h);
        
        if recording == 1
        file_h = fopen('rawdata_1.txt','a');
        fprintf(file_h,'%d %d %d \n', trialnum, pow1num, pow2num);
        fclose(file_h);
        end
        
        axes(handles.axes2);
        
     %   [hbo(1),hb(1)] = MBLL([0],[0],3,850,735);
      
     
     
        graph_h2 = plot(X,hbo,'r',X, hb,'b');
        grid on;
        %ylim([0, 1100]);
        xlabel('Time [ sec ]');
        ylabel('MBLL');
        title('MBLL');
        drawnow;
        
        
        if breaknow == 1
            break
        end  
        
    end
    
    fclose(s2);
    delete(instrfindall)           % 현재 연결되어 있는 시리얼포트들을 다 지웁니다.
    state = 0;
    breaknow = 0;
    
    set(graph_h1, 'visible', 'off');
    set(graph_h2, 'visible', 'off');
    
else
    %set(handles.btn1,'Measure','running','enable','off');
    
    set(graph_h1, 'visible', 'off');
    set(graph_h2, 'visible', 'off');
    state = 0;
    breaknow = 1;
    
end
    
    

    


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recording;
recording = 1;

