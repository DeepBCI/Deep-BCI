function samples = sample(d,cases)
% MULTINOMIAL/SAMPLE 
%   samples = sample(dist,cases) returns cases number of samples
%   from the multinomail.

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


if(nargin <=2)
if(nargin ==1)
    cases = 1;
end    

    rs = rand(1,cases);

rs = sort(rs);

cdfind = 1;

 cdf = reshape(d.ps,1,prod(size(d.ps)));
cdf  = cumsum(cdf);
sampinds = zeros(size(rs));
lcdf = length(cdf);

i=1;
for(cdfind=1:lcdf)
    while( i<=cases & cdf(cdfind)>rs(i) )
        sampinds(i) = cdfind;
        i = i+1;
    end
    if(i>cases)
        break;
    end
end
sampinds = sampinds(randperm(length(sampinds)));
if(d.dim == 1)
    samples = zeros(cases,1);
else
    samples = zeros(cases,ndims(d.ps));
end
cs = num2cell(samples,1);
[cs{:}] = ind2sub(size(d.ps),sampinds);
samples = cat(1,cs{:})';
else
    error('Too many arguments')
end
