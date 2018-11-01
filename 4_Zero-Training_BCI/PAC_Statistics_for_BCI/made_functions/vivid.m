% VIVID Creates a Personalized Vivid Colormap
%
%  VIVID(M,...) Creates a colormap with M colors
%  VIVID(MINMAX,...) Creates a colormap with a custom intensity range
%  VIVID(CLRS,...) Creates a colormap with custom colors
%  CMAP = VIVID(...) Exports the vivid colormap to a Mx3 matrix
%
%   Inputs:
%       M - (optional) an integer between 1 and 256 specifying the number
%           of colors in the colormap. Default is 128.
%       MINMAX - (optional) is a 1x2 vector with values between 0 and 1
%           representing the intensity range for the colors, which correspond
%           to black and white, respectively. Default is [0.15 0.85].
%       CLRS - (optional) either a Nx3 matrix of values between 0 and 1
%           representing the desired colors in the colormap
%               -or-
%           a string of characters that includes any combination of the
%           following letters:
%               'r' = red           'g' = green         'b' = blue
%               'y' = yellow        'c' = cyan          'm' = magenta
%               'o' = orange        'l' = lime green    'a' = aquamarine
%               's' = sky blue      'v' = violet        'p' = pink
%               'n' = navy blue     'f' = forest green
%               'k' or 'w' = black/white/grayscale
%
%   Outputs:
%       CMAP - an Mx3 colormap matrix
%
%   Example:
%       % Default Colormap
%       imagesc(sort(rand(200),'descend'));
%       colormap(vivid); colorbar
%
%   Example:
%       % Mapping With 256 Colors
%       imagesc(peaks(500))
%       colormap(vivid(256)); colorbar
%
%   Example:
%       % Mapping With Full Intensity Range
%       mesh(peaks(500))
%       colormap(vivid([0 1])); colorbar
%
%   Example:
%       % Mapping With Light Colors
%       mesh(peaks(500))
%       colormap(vivid([.5 1])); colorbar
%
%   Example:
%       % Mapping With Dark Colors
%       mesh(peaks(500))
%       colormap(vivid([0 .5])); colorbar
%
%   Example:
%       % Mapping With Custom Color Matrix
%       imagesc(peaks(500))
%       clrs = [.5 0 1; 0 .5 1; 0 1 .5; .5 1 0; 1 .5 0; 1 0 .5;];
%       colormap(vivid(clrs)); colorbar
%
%   Example:
%       % Mapping With Color String
%       imagesc(peaks(500))
%       colormap(vivid('pmvbscaglyor')); colorbar
%
%   Example:
%       % Colormap With Multiple Custom Settings
%       imagesc(sort(rand(300,100),'descend'));
%       colormap(vivid(64,[.1 .9],'bwr')); colorbar
%
%   Example:
%       % Topo Colormap
%       load topo;
%       imagesc(topo); axis xy; caxis([-6000 6000])
%       colormap(vivid('bg')); colorbar
%
% See also: jet, hsv, gray, hot, cold, copper, bone, fireice
%
% Author: Joseph Kirk
% Email: jdkirk630@gmail.com
% Release: 1.1
% Date: 12/09/11
function cmap = vivid(varargin)


% Default Color Spectrum
clrs = [1 0 1;.5 0 1;0 0 1;0 1 1    % Magenta, Violet, Blue, Cyan
    0 1 0;1 1 0;1 .5 0;1 0 0];      % Green, Yellow, Orange, Red

% Default Min/Max Intensity Range
minmax = [0.15 0.85];

% Default Colormap Size
m = 256;

% Process Inputs
for var = varargin
    input = var{1};
    if ischar(input)
        nColors = length(input(:));
        colorMat = zeros(nColors,3);
        c = 0;
        for k = 1:nColors
            c = c + 1;
            switch lower(input(k))
                case 'r', colorMat(c,:) = [1 0 0];  % red
                case 'g', colorMat(c,:) = [0 1 0];  % green
                case 'b', colorMat(c,:) = [0 0 1];  % blue
                case 'y', colorMat(c,:) = [1 1 0];  % yellow
                case 'c', colorMat(c,:) = [0 1 1];  % cyan
                case 'm', colorMat(c,:) = [1 0 1];  % magenta
                case 'p', colorMat(c,:) = [1 0 .5]; % pink
                case 'o', colorMat(c,:) = [1 .5 0]; % orange
                case 'l', colorMat(c,:) = [.5 1 0]; % lime green
                case 'a', colorMat(c,:) = [0 1 .5]; % aquamarine
                case 's', colorMat(c,:) = [0 .5 1]; % sky blue
                case 'v', colorMat(c,:) = [.5 0 1]; % violet
                case 'f', colorMat(c,:) = [0 .5 0]; % forest green
                case 'n', colorMat(c,:) = [0 0 .5]; % navy
                case {'k','w'}, colorMat(c,:) = [.5 .5 .5]; % grayscale
                otherwise, c = c - 1;
                    fprintf('Warning: Input character [%s] is not a recognized color ...\n',input(k));
            end
        end
        colorMat = colorMat(1:c,:);
        if ~isempty(colorMat)
            clrs = colorMat;
        end
    elseif isnumeric(input)
        if isscalar(input)
            m = max(1,min(256,round(real(input))));
        elseif size(input,2) == 3
            clrs = max(0,min(1,real(input)));
        elseif length(input) == 2
            minmax = max(0,min(1,real(input)));
        end
    end
end

% Calculate Parameters
nc = size(clrs,1);  % number of spectrum colors
ns = ceil(m/nc);    % number of shades per color
n = nc*ns;
d = n - m;

% Scale Intensity
sup = 2*minmax;
sub = 2*minmax - 1;
if ns == 1
    high = repmat(min(1,mean(sup)),[1 nc 3]);
    low = repmat(max(0,mean(sub)),[1 nc 3]);
else
    high = repmat(min(1,linspace(sup(1),sup(2),ns))',[1 nc 3]);
    low = repmat(max(0,linspace(sub(1),sub(2),ns))',[1 nc 3]);
end

% Determine Color Spectrum
rgb = repmat(reshape(clrs,1,nc,3),ns,1);
map = rgb.*high + (1-rgb).*low;

% Obtain Color Map
cmap = reshape(map,n,3,1);
cmap(1:ns:d*ns,:) = [];

