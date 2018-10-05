function handle = arrowh(x,y,clr,ArSize,Where)
%  ARROWH   Draws a solid 2D arrow head in current plot.
%
%	 ARROWH(X,Y,COLOR,SIZE,LOCATION) draws a  solid arrow  head into
%	 the current plot to indicate a direction.  X and Y must contain
%	 a pair of x and y coordinates ([x1 x2],[y1 y2]) of two points:
%
%	 The first  point is only used to tell  (in conjunction with the
%	 second one)  the direction  and orientation of  the arrow -- it
%	 will point from the first towards the second.
%
%	 The head of the arrow  will be located in the second point.  An
%	 example of use is	plot([0 2],[0 4]); ARROWH([0 1],[0 2],'b')
%
%	 You may also give  two vectors of same length > 2.  The routine
%	 will then choose two consecutive points from "about" the middle
%	 of each vectors.  Useful if you  don't want to worry  each time
%	 about  where to  put the arrows on  a trajectory.  If x1 and x2
%	 are the vectors x1(t) and x2(t), simply put   ARROWH(x1,x2,'r')
%	 to have the right  direction indicated in your x2 = f(x1) phase
%	 plane.
%
%            (x2,y2)
%            --o
%            \ |
%	            \|
%
%
%		  o
%	  (x1,y1)
%
%	 Please note  that the following  optional arguments  need -- if
%	 you want  to use them -- to  be given in that exact order.  You
%	 may pass on empty vectors "[]" to skip arguments you don't want
%	 to set (if you want to access "later" arguments...).
%
%	 The COLOR argument is quite the same as for plots,  i.e. either
%  a string like  'r' or an RGB value vector like  [1 0 0]. If you
%  only want the outlines of the head  (in other words a non-solid
%  arrow head), prefix the color string by 'e' or the color vector
%  by 0, e.g. to get only a red outline use 'er' or [0 1 0 0].
%
%	 The SIZE argument allows you to tune the size of the arrows. If
%	 SIZE is a scalar, it scales the arrow proportionally.  SIZE can
%	 also be  a two element vector,  where the first element  is the
%	 overall  scale (in percent),  the second one controls the width
%	 of the arrow head (again, in percent).
%
%	 The LOCAITON argument can be used to tweak the position  of the
%  arrow head.  If a time series of x and y coordinates are given,
%  you can use this argument  to place the arrow head for instance
%  at 20% along the line.  It can be a vector, if you want to have
%  more than one arrow head drawn.
%
%	 Both SIZE and LOCATION arguments must also be given in percent,
%	 where 100 means standard size, 50 means half size, respectively
%	 100 means end of the vector, 0 beginning of it. Note that those
%	 "locations" correspond to the cardinal position "inside" the
%	 vector, in other words the "index-wise" position.
%
%	 This little tool is mainely intended  to be used for indicating
%	 "directions" on trajectories -- just give two consecutive times
%	 and the corresponding values of a flux and the proper direction
%	 of the trajectory will be shown on the plot.  You may also pass
%	 on two solution vectors, as described above.
%
%	 Note, that the arrow  heads only look good in the original axis
%	 settings (as in when the routine was actually started).  If you
%	 zoom in afterwards, the triangle will get distorted.
%
%  HANDLES = ARROWH(...)  will give you a vector with  the handles
%  to the patches created by this function  (if you want to modify
%  them later on, for instance).
%
%	 Examples of use:
% 	 x1 = [0:.2:2]; x2 = [0:.2:2]; plot(x1,x2); hold on;
% 	 arrowh(x1,x2,'r',[],20);            % passing entire vectors
% 	 arrowh([0 1],[0 1],'eb',[300,75]);  % passing 2 points
% 	 arrowh([0 1],[0 1],'eb',[300,75],25); % head closer to (x1,y1)

%	 Author:     Florian Knorn
%	 Email:      florian@knorn.org
%	 Version:    1.14
%	 Filedate:   Jun 18th, 2008
%
%	 History:    1.14 - LOCATION now also works with lines
%              1.13 - Allow for non-solid arrow heads
%              1.12 - Return handle(s) of created patches
%              1.11 - Possibility to change width
%	             1.10 - Buxfix
%	             1.09 - Possibility to chose *several* locations
%	             1.08 - Possibility to chose location
%	             1.07 - Choice of color
%	             1.06 - Bug fixes
%	             1.00 - Release
%
%	 ToDos:      - Keep proportions when zooming or resizing; has to
%	               be done with callback functions, I guess.
%
%	 Bugs:       None discovered yet, those discovered were fixed
%
%	 Thanks:     Thanks  also  to Oskar Vivero  for using  my humble
%	             little program in his great MIMO-Toolbox.
%
%	 If you have  suggestions for  this program,  if it doesn't work
%	 for your "situation" or if you change something in it -- please
%	 send me an email!  This is my very  first "public" program  and
%	 I'd  like to  improve it where  I can -- your  help is  kindely
%	 appreciated! Thank you!


%-- errors
if nargin < 2
	error('Please give enough coordinates !');
end
if (length(x) < 2) || (length(y) < 2),
	error('X and Y vectors must each have "length" >= 2 !');
end
if (x(1) == x(2)) && (y(1) == y(2)),
	error('Points superimposed - cannot determine direction !');
end
if nargin <= 2
	clr = 'b';
end
if nargin <= 3
	ArSize = [100,100];
end

handle = [];


%-- check if variables left empty, deal width ArSize and Color
if isempty(clr)
	clr = 'b'; nonsolid = false;
elseif ischar(clr)
	if strncmp('e',clr,1) % for non-solid arrow heads
		nonsolid = true; clr = clr(2);
	else
		nonsolid = false;
	end
elseif isvector(clr)
	if length(clr) == 4 && clr(1) == 0  % for non-solid arrow heads
		nonsolid = true;
		clr = clr(2:end);
	else
		nonsolid = false;
	end
else
	error('COLOR argument of wrong type (must be either char or vector)');
end

if nargin <= 4
	if (length(x) == length(y)) && (length(x) == 2)
		Where = 100;
	else
		Where = 50;
	end
end

if isempty(ArSize)
	ArSize = [100,100];
end
if length(ArSize) == 2
	ArWidth = 0.75*ArSize(2)/100; % .75 to make arrows it a bit slimmer
else
	ArWidth = 0.75;
end
ArSize = ArSize(1);

%-- determine and remember the hold status, toggle if necessary
if ishold,
	WasHold = 1;
else
	WasHold = 0;
	hold on;
end

%-- start for-loop in case several arrows are wanted
for Loop = 1:length(Where),

	%-- if vectors "longer" then 2 are given we're dealing with time series
	if (length(x) == length(y)) && (length(x) > 2),
		j = floor(length(x)*Where(Loop)/100); %-- determine that location
		if j >= length(x), j = length(x) - 1; end
		if j == 0, j = 1; end
		x1 = x(j); x2 = x(j+1); y1 = y(j); y2 = y(j+1);

	else %-- just two points given - take those
		x1 = x(1); x2 = (1-Where/100)*x(1)+Where/100*x(2);
		y1 = y(1); y2 = (1-Where/100)*y(1)+Where/100*y(2);
	end


	%-- get axe ranges and their norm
	OriginalAxis = axis;
	Xextend = abs(OriginalAxis(2)-OriginalAxis(1));
	Yextend = abs(OriginalAxis(4)-OriginalAxis(3));

	%-- determine angle for the rotation of the triangle
	if x2 == x1, %-- line vertical, no need to calculate slope
		if y2 > y1,
			p = pi/2;
		else
			p= -pi/2;
		end
	else %-- line not vertical, go ahead and calculate slope
		%-- using normed differences (looks better like that)
		m = ( (y2 - y1)/Yextend ) / ( (x2 - x1)/Xextend );
		if x2 > x1, %-- now calculate the resulting angle
			p = atan(m);
		else
			p = atan(m) + pi;
		end
	end

	%-- the arrow is made of a transformed "template triangle".
	%-- it will be created, rotated, moved, resized and shifted.

	%-- the template triangle (it points "east", centered in (0,0)):
	xt = [1	-sin(pi/6)	-sin(pi/6)];
	yt = ArWidth*[0	 cos(pi/6)	-cos(pi/6)];

	%-- rotate it by the angle determined above:
	xd = []; yd = [];
	for i=1:3
		xd(i) = cos(p)*xt(i) - sin(p)*yt(i);
		yd(i) = sin(p)*xt(i) + cos(p)*yt(i);
	end

	%-- move the triangle so that its "head" lays in (0,0):
	xd = xd - cos(p);
	yd = yd - sin(p);

	%-- stretch/deform the triangle to look good on the current axes:
	xd = xd*Xextend*ArSize/10000;
	yd = yd*Yextend*ArSize/10000;

	%-- move the triangle to the location where it's needed
	xd = xd + x2;
	yd = yd + y2;

	%-- draw the actual triangle
	handle(Loop) = patch(xd,yd,clr,'EdgeColor',clr);
	if nonsolid, set(handle(Loop),'facecolor','none'); end
end % Loops

%-- restore original axe ranges and hold status
axis(OriginalAxis);
if ~WasHold,
	hold off
end

%-- work done. good bye.