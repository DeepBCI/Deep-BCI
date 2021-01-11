function [ffl2,ffr2,fft2,ddo2] =ega(mi2,lcsp2,rcsp2,tcsp2,dif_t2)
[ko2,kko2]=sort(mi2,'descend');
ffl2=[lcsp2(:,:,kko2(1)),lcsp2(:,:,kko2(2))];
ffr2=[rcsp2(:,:,kko2(1)),rcsp2(:,:,kko2(2))];
fft2=[tcsp2(:,:,kko2(1)),tcsp2(:,:,kko2(2))];
ddo2=(dif_t2(kko2(1))+dif_t2(kko2(2)))/2;
% ddo2=dif_t2(kko2(1));
end