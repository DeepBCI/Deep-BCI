function [] = topoplot_connect(displayStruct, loc_file)
% NOTE: The cartoon head is drawn using code in topoplot.m file from EEGLAB
% v6.01b (http://sccn.ucsd.edu/eeglab/). 
%
% Reference:
% Delorme A & Makeig S (2004) EEGLAB: an open source toolbox for analysis
% of single-trial EEG dynamics. Journal of Neuroscience Methods 134:9-21 
%
% Usage:
%
% >> topoplot_connect(ds, EEG.chanlocs);
%
% *ds* is the display strcture with the folowing fields:
%
% * *ds.chanPairs* (required) - N x 2 matrix, with N being the number of 
% connected channel pairs. For example, ds.chanPairs = [7, 12; 13 20]; 
% specifies two connected channel pairs (7, 12) and (13, 20).
% * *ds.connectStrength* (optional) - N x 1 matrix, a vector specifying
% connection strengths. If unspecified, then the connections will be
% rendered in a color at the center of the current colormap.
% * *ds.connectStrengthLimits* (optional) - 1 x 2 matrix specifying minimum
% and maximum values of connection strengths to display. If it is not 
% specified, then the minimum and maximum values from ds.connectStrength 
% are used.
%
% EEG.chanlocs is a structure specifying channel locations (or an locs
% filename)
%
% For comments and/or suggestions, please send me an email at
% praneeth@mit.edu
%

BACKCOLOR = [1 1 1];  % EEGLAB standard
rmax = 0.5;             % actual head radius - Don't change this!
CIRCGRID   = 201;       % number of angles to use in drawing circles
AXHEADFAC = 1.3;        % head to axes scaling factor
HEADCOLOR = [0 0 0];    % default head color (black)
EMARKER = '.';          % mark electrode locations with small disks
ECOLOR = [0 0 0];       % default electrode color = black
EMARKERSIZE = [];       % default depends on number of electrodes, set in code
EMARKERLINEWIDTH = 1;   % default edge linewidth for emarkers
HLINEWIDTH = 1.7;         % default linewidth for head, nose, ears
HEADRINGWIDTH    = .007;% width of the cartoon head ring

%
%%%%%%%%%%%%%%%%%%%% Read the channel location information %%%%%%%%%%%%%%%%%%%%%%%%
%
if ischar(loc_file)
    [~, ~, Th Rd indices] = readlocs( loc_file,'filetype','loc');
elseif isstruct(loc_file) % a locs struct
    [~, ~, Th Rd indices] = readlocs( loc_file );
    % Note: Th and Rd correspond to indices channels-with-coordinates only
else
    error('loc_file must be a EEG.locs struct or locs filename');
end
Th = pi/180*Th;                              % convert degrees to radians
plotchans = indices;
%
%%%%%%%%%%%%%%%%%%% remove infinite and NaN values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

[x,y]     = pol2cart(Th,Rd);  % transform electrode locations from polar to cartesian coordinates
plotchans = abs(plotchans);   % reverse indicated channel polarities
Rd        = Rd(plotchans);
x         = x(plotchans);
y         = y(plotchans);
plotrad = min(1.0,max(Rd)*1.02);            % default: just outside the outermost electrode location
plotrad = max(plotrad,0.5);                 % default: plot out to the 0.5 head boundary
headrad = rmax;
pltchans = find(Rd <= plotrad); % plot channels inside plotting circle
x     = x(pltchans);
y     = y(pltchans);
squeezefac = rmax/plotrad;
x    = x*squeezefac;
y    = y*squeezefac;

%
%%%%%%%%%%%%%%%%%%%%%%% Draw blank head %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
cla
hold on
set(gca,'Xlim',[-rmax rmax]*AXHEADFAC,'Ylim',[-rmax rmax]*AXHEADFAC)

%
%%%%%%%%%%%%%%%%%%% Plot filled ring to mask jagged grid boundary %%%%%%%%%%%%%%%%%%%%%%%%%%%
%
hwidth = HEADRINGWIDTH;                   % width of head ring
hin  = squeezefac*headrad*(1- hwidth/2);  % inner head ring radius
circ = linspace(0,2*pi,CIRCGRID);
rx = sin(circ);
ry = cos(circ);

%
%%%%%%%%%%%%%%%%%%%%%%%%% Plot cartoon head, ears, nose %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
headx = [[rx(:)' rx(1) ]*(hin+hwidth)  [rx(:)' rx(1)]*hin];
heady = [[ry(:)' ry(1) ]*(hin+hwidth)  [ry(:)' ry(1)]*hin];

patch(headx,heady,ones(size(headx)),HEADCOLOR,'edgecolor',HEADCOLOR); hold on

%%%%%%%%%%%%%%%%%%% Plot ears and nose %%%%%%%%%%%%%%%%%%%%%%%%%%%
%
base  = rmax-.0046;
basex = 0.18*rmax;                   % nose width
tip   = 1.15*rmax;
tiphw = .04*rmax;                    % nose tip half width
tipr  = .01*rmax;                    % nose tip rounding
q = .04; % ear lengthening
EarX  = [.497-.005  .510  .518  .5299 .5419  .54    .547   .532   .510   .489-.005]; % rmax = 0.5
EarY  = [q+.0555 q+.0775 q+.0783 q+.0746 q+.0555 -.0055 -.0932 -.1313 -.1384 -.1199];
sf    = headrad/plotrad;

plot3([basex;tiphw;0;-tiphw;-basex]*sf,[base;tip-tipr;tip;tip-tipr;base]*sf,...
    2*ones(size([basex;tiphw;0;-tiphw;-basex])),...
    'Color',HEADCOLOR,'LineWidth',HLINEWIDTH);                 % plot nose
plot3(EarX*sf,EarY*sf,2*ones(size(EarX)),'color',HEADCOLOR,'LineWidth',HLINEWIDTH)    % plot left ear
plot3(-EarX*sf,EarY*sf,2*ones(size(EarY)),'color',HEADCOLOR,'LineWidth',HLINEWIDTH)   % plot right ear

%
% %%%%%%%%%%%%%%%%%%% Show electrode information %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
plotax = gca;
axis square                                           % make plotax square
axis off

pos = get(gca,'position');
set(plotax,'position',pos);

xlm = get(gca,'xlim');
set(plotax,'xlim',xlm);

ylm = get(gca,'ylim');
set(plotax,'ylim',ylm);                               % copy position and axis limits again


if isempty(EMARKERSIZE)
    EMARKERSIZE = 10;
    if length(y)>=32
        EMARKERSIZE = 8;
    elseif length(y)>=48
        EMARKERSIZE = 6;
    elseif length(y)>=64
        EMARKERSIZE = 5;
    elseif length(y)>=80
        EMARKERSIZE = 4;
    elseif length(y)>=100
        EMARKERSIZE = 3;
    elseif length(y)>=128
        EMARKERSIZE = 3;
    elseif length(y)>=160
        EMARKERSIZE = 3;
    end
end
%
%%%%%%%%%%%%%%%%%%%%%%%% Mark electrode locations only %%%%%%%%%%%%%%%%%%%%%%%%%%
%
ELECTRODE_HEIGHT = 2.1;  % z value for plotting electrode information (above the surf)
plot3(y,x,ones(size(x))*ELECTRODE_HEIGHT,...
    EMARKER,'Color',ECOLOR,'markersize',EMARKERSIZE,'linewidth',EMARKERLINEWIDTH);

% praneeth - starts
numChanPairs = size(displayStruct.chanPairs, 1);
cM = colormap;
if ~isfield(displayStruct, 'connectStrength')
    cmapPos = ceil(size(cM, 1)/2)*ones(size(displayStruct.chanPairs, 1), 1);
else
    if ~isfield(displayStruct, 'connectStrengthLimits')
        displayStruct.connectStrengthLimits = [min(displayStruct.connectStrength), max(displayStruct.connectStrength)];
    end
    xp = displayStruct.connectStrengthLimits(1);
    yp = displayStruct.connectStrengthLimits(2);
    displayStruct.connectStrength(displayStruct.connectStrength < xp) = xp;
    displayStruct.connectStrength(displayStruct.connectStrength > yp) = yp;
    if xp == yp
        cmapPos = ceil(size(cM, 1)/2)*ones(size(displayStruct.chanPairs, 1), 1);
    else
        cmapPos = round((displayStruct.connectStrength - xp)/(yp - xp)*(size(cM, 1) - 1) + 1);
    end
end

for kk = 1:numChanPairs
    plot3(y(displayStruct.chanPairs(kk, :)), x(displayStruct.chanPairs(kk, :)), [ELECTRODE_HEIGHT, ELECTRODE_HEIGHT], 'LineWidth', 2, 'Color', cM(cmapPos(kk), :));
end
% praneeth - ends
set(gcf, 'color', BACKCOLOR);
return;