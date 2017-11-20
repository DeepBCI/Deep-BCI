function d = gaussian(m,c)
%GAUSSIAN Gaussian distribution class constructor.
%   d = GAUSSIAN(m,c) creates a Gaussian distribution with
%   mean m and covariance matrix c; may also be called
%   like d = GAUSSIAN(p) where p is a gaussian to be copied

% Copyright October, 2006, Brown University, Providence, RI. 
% All Rights Reserved 

% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a commercial
% product is hereby granted without fee, provided that the above copyright
% notice appear in all copies and that both that copyright notice and this
% permission notice appear in supporting documentation, and that the name of
% Brown University not be used in advertising or publicity pertaining to
% distribution of the software without specific, written prior permission. 

% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE. IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY
% SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
% RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
% CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
% CONNECTION WITH THE USE.

% Author: Frank Wood fwood@cs.brown.edu

if nargin == 1
    if(isa(m,'gaussian'))
        d.m = m.m;
        d.c = m.c;
        d = class(d,'gaussian',dist);
    else
        d.m = 0;
        d.c = 1;
        bc = distribution();
        d = class(d,'gaussian',bc);
        d = fit(d,m);

    end
elseif nargin == 2
    d.m = m;
    d.c = c;
    bc = distribution();
    d = class(d,'gaussian',bc);
else
    error;
end
