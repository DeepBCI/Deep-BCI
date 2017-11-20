function g = fit(d,data)
% MULTINOMIAL/FIT  Fit multinomial to a list of integers.  This function
% assumes that the integers are contiguous and start from 1 (i.e. 1...N)
% this creates a multinomial with N bins where each probability is the 
% normalized count of that integer

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


dim = size(data,2);
uvals = cell(dim,1);
nuvals = zeros(dim,1)';

for(d = 1:dim)
uvals{d} = sort(unique(data(:,d)));
nuvals(d) = max(uvals{d}); % could be length, max better for neural data
end

counts = zeros(nuvals);

inds = num2cell(data,1);
cii = sub2ind(size(counts),inds{:});
[N,I] = hist(cii,1:prod(size(counts)));

counts(I) = N;

counts = counts./length(data);
g = multinomial(counts);
