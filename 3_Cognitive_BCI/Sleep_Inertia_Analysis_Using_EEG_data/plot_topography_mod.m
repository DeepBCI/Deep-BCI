% Function modify by hn_jo@korea.ac.kr 
% last update 2022. 05. 13         

% 2022. 05. 13 / Add Bar min, max factor - minlim, maxlim (list, double) 
% 2022. 05. 13 / Add pvalue factor - pvalues (list, 0 or 1)
% 2022. 05. 13 / Add pvalue plot factor - plot_pvalues (bool)
% 2022. 05. 13~ / Need to add Bonferroni correction

% ----------------------------------------------------------------------- %
% Function 'plot_topography' plots a topographical map of the head over the
% desired points given by 'ch_list' and their assigned 'values'.          %
%                                                                         %
%   Input parameters:                                                     %
%       - ch_list:      Channel list in cell-array. Use the string 'all'  %
%                       for displaying all channels available. Note that  %
%                       'z' indicator should be in lower case.            %
%                       Example: ch_list = {'Fpz','Fz','Cz','Pz','Oz'};   %
%       - values:       Numeric vector that contains the values assigned  %
%                       to each channel.                                  %
%       - make_contour: (Optional, default: false) Boolean that controls if
%                       the contour lines should be plotted.              %
%       - system:       (Optional) Measurement system as a string:        %
%           * '10-20':  (Default) 10-20 System, 81 electrodes are available
%           * '10-10':  10-10 system, 47 electrodes are available.        %
%           * 'yokogawa': MEG system by Yokogawa (in testing).            %
%           * table:    Specify a table containing the custom locations,  %
%                       following the structure that is detailed below.   %
%           * path:     Specify a path of a .mat file containing custom   %
%                       locations. The file must have a MATLAB table named%
%                       'locations', with 3 mandatory columns:            %
%                           - labels: contains the name of the electrodes.%
%                           - theta: angle of polar coordinate (in degrees)
%                           - radius: radius of polar coordinate (0-1).   %
%                       Example of 'locations' table:                     %
%                           labels      theta       radius                %
%                       --------------------------------------            %
%                           'Fpz'       0           0.511                 %
%                           'Cz'        90          0                     %
%                           'Oz'        180         0.511                 %
%       - plot_channels:(Optional, default: false) Boolean that controls if
%                       the electrodes should be plotted.                 %
%       - plot_clabels: (Optional, default: false) Boolean that controls if
%                       the text labels of each electrode should be plotted.
%       - INTERP_POINTS:(Optional, default: 1000) No. of interpolation    %
%                       points. The lower N, the lower resolution and     %
%                       faster computation.                               %
%                                                                         %
%   Output variables:                                                     %
%       - h:            Figure handle.                                    %
%                                                                         %
%   Notes:                                                                %
%       - This code was intended to have a small number of input parameters
%       to favor the readability. Therefore, feel free to modify aspects  %
%       such as electrode markers, line widths, colormaps and so on. The  %
%       following lines indicate the key points to modify these aspects:  %
%           * Line 112: Global parameters of number of interpolation      %
%           points or head radius. INTERP_POINTS is set to 1000, if the   %
%           number is increased, the output will be smoother but it will  %
%           take more time. HEAD_RADIUS is fixed to the optimal value in  %
%           order to be anatomically correct if 10-20 system is used.     %
%           * Line 139: Type of interpolation, now 'v4' is used in order to
%           interpolate the surface over a rectangle from -1 to 1.        %
%           * Line 145: Interpolation pcolor.                             %
%           * Line 155: Head plot.                                        %
%           * Line 166: Nose plot.                                        %
%           * Line 183: Ear plots.                                        %
%           * Line 187: Electrode plots.                                  %
% ----------------------------------------------------------------------- %
%   Versions:                                                             %
%       - 1.0:          (18/01/2019) Original script.                     %
%       - 1.1:          (23/01/2019) Yokogawa system added.               %
%       - 1.2:          (04/02/2019) Plot channels added.                 %
%       - 1.3:          (11/09/2019) K-nearest neighbors interpolation.   %
%       - 1.4:          (21/10/2019) Now locations can be directly speci- %
%                       fied as an input table and channel points can be  %
%                       hidden. No. of interp. points may be also passed as
%                       a parameter.                                      %
%       - 1.5:          (23/11/2020) Locations in uppercase.              %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:      1.4.                                              %
%       - Author:       V. Martinez-Cagigal                               %
%       - Date:         23/11/2020                                        %
% ----------------------------------------------------------------------- %
%   Example of use:                                                       %
%       plot_topography('all', rand(1,81));                               %
% ----------------------------------------------------------------------- %
function h = plot_topography(ch_list, values, ...
    maxlim, minlim, ...
    pvalues, plot_pvalues, make_contour, system, ...
    plot_channels, plot_clabels, INTERP_POINTS)

    % Error detection
    if nargin < 2, error('[plot_topography] Not enough parameters.');
    else
        if ~iscell(ch_list) && ~ischar(ch_list)
            error('[plot_topography] ch_list must be "all" or a cell array.');
        end
        if ~isnumeric(values)
            error('[plot_topography] values must be a numeric vector.');
        end
    end
%     if nargin < 3, maxlim = 100; end
%     if nargin < 4, minlim = -100; end
    if nargin < 5, pvalues = []; end
    if nargin < 6, plot_pvalues = true; end
    if nargin < 7, make_contour = true; 
    else
        if make_contour~=1 && make_contour~=0
            error('[plot_topography] make_contour must be a boolean (true or false).');
        end
    end
    if nargin < 8, system = '10-20';
    else
        if ~ischar(system) && ~istable(system)
            error('[plot_topography] system must be a string or a table.');
        end
    end
    if nargin < 9, plot_channels = true;
    else
        if plot_channels~=1 && plot_channels~=0
            error('[plot_topography] plot_channels must be a boolean (true or false).');
        end
    end
    if nargin < 10, plot_clabels = false;
    else
        if plot_clabels~=1 && plot_clabels~=0
            error('[plot_topography] plot_clabels must be a boolean (true or false).');
        end
    end
    if nargin < 11, INTERP_POINTS = 1000;
    else
        if ~isnumeric(INTERP_POINTS)
            error('[plot_topography] N must be an integer.');
        else
            if mod(INTERP_POINTS,1) ~= 0
                error('[plot_topography] N must be an integer.');
            end
        end
    end
    
    % Loading electrode locations
    if ischar(system)
        switch system
            case '10-20'
                % 10-20 system
                load('Standard_10-20_81ch.mat', 'locations')
%                 load('Custom_10-20_60ch.mat', 'locations');
            case '10-10'
                % 10-10 system
                load('Standard_10-10_47ch.mat', 'locations');
            case 'yokogawa'
                % Yokogawa MEG system
                load('MEG_Yokogawa-440ag.mat', 'locations');
            otherwise
                % Custom path
                load(system, 'locations');
        end
    else
        % Custom table
        locations = system;
    end
    
    % Finding the desired electrodes
    ch_list = upper(ch_list);
    if ~iscell(ch_list)
        if strcmp(ch_list,'all')
            idx = 1:length(locations.labels);
            if length(values) ~= length(idx)
                error('[plot_topography] There must be a value for each of the %i channels.', length(idx));
            end
        else, error('[plot_topography] ch_list must be "all" or a cell array.');
        end
    else
        if length(values) ~= length(ch_list)
            error('[plot_topography] values must have the same length as ch_list.');
        end
        idx = NaN(length(ch_list),1);
        for ch = 1:length(ch_list)
            if isempty(find(strcmp(locations.labels,ch_list{ch})))
%                 disp(locations.labels);
%                 disp(ch_list{ch});
                warning('[plot_topography] Cannot find the %s electrode.',ch_list{ch});
                ch_list{ch} = [];
                values(ch)  = [];
                idx(ch)     = [];
            else
                idx(ch) = find(strcmp(locations.labels,ch_list{ch}));
            end
        end
    end
    values = values(:);
    
    % Global parameters
    %   Note: Head radius should be set as 0.6388 if the 10-20 system is used.
    %   This number was calculated taking into account that the distance from Fpz
    %   to Oz is d=2*0.511. Thus, if the circle head must cross the nasion and
    %   the inion, it should be set at 5d/8 = 0.6388.
    %   Note2: When the number of interpolation points rises, the plots become
    %   smoother and more accurate, however, computational time also rises.
    HEAD_RADIUS     = 5*2*0.511/8;  % 1/2  of the nasion-inion distance
    HEAD_EXTRA      = 1*2*0.511/8;  % 1/10 of the nasion-inion distance
    k = 4;                          % Number of nearest neighbors for interpolation
    
    % Interpolating input data
        % Creating the rectangle grid (-1,1)
        [ch_x, ch_y] = pol2cart((pi/180).*((-1).*locations.theta(idx)+90), ...
                                locations.radius(idx));     % X, Y channel coords
        % Points out of the head to reach more natural interpolation
        r_ext_points = 1.2;
        [add_x, add_y] = pol2cart(0:pi/4:7*pi/4,r_ext_points*ones(1,8));
        linear_grid = linspace(-r_ext_points,r_ext_points,INTERP_POINTS);         % Linear grid (-1,1)
        [interp_x, interp_y] = meshgrid(linear_grid, linear_grid);
        
        % Interpolate and create the mask
        outer_rho = max(locations.radius(idx));
        if outer_rho > HEAD_RADIUS, mask_radius = outer_rho + HEAD_EXTRA;
        else,                       mask_radius = HEAD_RADIUS;
        end
        mask = (sqrt(interp_x.^2 + interp_y.^2) <= mask_radius); 
        add_values = compute_nearest_values([add_x(:), add_y(:)], [ch_x(:), ch_y(:)], values(:), k);
        interp_z = griddata([ch_x(:); add_x(:)], [ch_y(:); add_y(:)], [values; add_values(:)], interp_x, interp_y, 'natural');
        interp_z(mask == 0) = NaN;

        % Plotting the final interpolation
        pcolor(interp_x, interp_y, interp_z);
        colormap 'jet';
        shading interp;
        hold on;
        
        % Contour => 윤곽선 / 등고선
        if make_contour
            [~, hfigc] = contour(interp_x, interp_y, interp_z); 
            set(hfigc, 'LineWidth',0.5, 'Color', [0.2 0.2 0.2]); 
            hold on;
        end

    % Plotting the head limits as a circle         
    head_rho    = HEAD_RADIUS;                      % Head radius
    if strcmp(system,'yokogawa'), head_rho = 0.45; end
    head_theta  = linspace(0,2*pi,INTERP_POINTS);   % From 0 to 360Âº
    head_x      = head_rho.*cos(head_theta);        % Cartesian X of the head
    head_y      = head_rho.*sin(head_theta);        % Cartesian Y of the head
    plot(head_x, head_y, 'Color', 'k', 'LineWidth',4);
    hold on;

    % Plotting the nose
    nt = 0.15;      % Half-nose width (in percentage of pi/2)
    nr = 0.22;      % Nose length (in radius units)
    nose_rho   = [head_rho, head_rho+head_rho*nr, head_rho];
    nose_theta = [(pi/2)+(nt*pi/2), pi/2, (pi/2)-(nt*pi/2)];
    nose_x     = nose_rho.*cos(nose_theta);
    nose_y     = nose_rho.*sin(nose_theta);
    plot(nose_x, nose_y, 'Color', 'k', 'LineWidth',4);
    hold on;

    % Plotting the ears as ellipses
    ellipse_a = 0.08;                               % Horizontal exentricity
    ellipse_b = 0.16;                               % Vertical exentricity
    ear_angle = 0.9*pi/8;                           % Mask angle
    offset    = 0.05*HEAD_RADIUS;                   % Ear offset
    ear_rho   = @(ear_theta) 1./(sqrt(((cos(ear_theta).^2)./(ellipse_a^2)) ...
        +((sin(ear_theta).^2)./(ellipse_b^2))));    % Ellipse formula in polar coords
    ear_theta_right = linspace(-pi/2-ear_angle,pi/2+ear_angle,INTERP_POINTS);
    ear_theta_left  = linspace(pi/2-ear_angle,3*pi/2+ear_angle,INTERP_POINTS);
    ear_x_right = ear_rho(ear_theta_right).*cos(ear_theta_right);          
    ear_y_right = ear_rho(ear_theta_right).*sin(ear_theta_right); 
    ear_x_left  = ear_rho(ear_theta_left).*cos(ear_theta_left);         
    ear_y_left  = ear_rho(ear_theta_left).*sin(ear_theta_left); 
    plot(ear_x_right+head_rho+offset, ear_y_right, 'Color', 'k', 'LineWidth',4); hold on;
    plot(ear_x_left-head_rho-offset, ear_y_left, 'Color', 'k', 'LineWidth',4); hold on;

    % Plotting the electrodes
    % [ch_x, ch_y] = pol2cart((pi/180).*(locations.theta(idx)+90), locations.radius(idx));
    if plot_channels, he = scatter(ch_x, ch_y, 60,'.','k', 'LineWidth',0.5); end
    % plot p-values
    if plot_pvalues, he = scatter(ch_x(pvalues), ch_y(pvalues), 60,'*','w', 'LineWidth',1.2); end
    if plot_clabels, text(ch_x, ch_y, ch_list); end
    if strcmp(system,'yokogawa'), delete(he); plot(ch_x, ch_y, '.k'); end
    
    % Last considerations
    max_height = max([max(nose_y), mask_radius]);
    min_height = -mask_radius;
    max_width  = max([max(ear_x_right+head_rho+offset), mask_radius]);
    min_width  = -max_width;
    L = max([min_height, max_height, min_width, max_width]);
    xlim([-L, L]);
    ylim([-L, L]);  
    
    c = colorbar;
%     c.Label.String = 't-value'; 
    c.Label.FontSize = 12;
    if nargin > 5, caxis([minlim maxlim]); end
    axis square;
    axis off;
    hold off;
    h = gcf;
end

% This function compute the mean values of the k-nearest neighbors
%   - coor_add:     XY coordinates of the virtual electrodes
%   - coor_neigh:   XY coordinates of the real electrodes
%   - val_neigh:    Values of the real electrodes
%   - k:            Number of neighbors to consider
function add_val = compute_nearest_values(coor_add, coor_neigh, val_neigh, k)
    
    add_val = NaN(size(coor_add,1),1);
    L = length(add_val);
    
    for i = 1:L
        % Distances between the added electrode and the original ones
        target = repmat(coor_add(i,:),size(coor_neigh,1),1);
        d = sqrt(sum((target-coor_neigh).^2,2));
        
        % K-nearest neighbors
        [~, idx] = sort(d,'ascend');
        idx = idx(2:1+k);
        
        % Final value as the mean value of the k-nearest neighbors
        add_val(i) = mean(val_neigh(idx));
    end
    
end