clear all
close all
clc

load('D:\Project\2021\NIRS\20210203\data\Discription');
[opy opx] = find(op_loc>0);
subdir_list = {'VP00','VP01','VP02','VP03','VP04','VP05','VP06','VP07','VP09','VP011','VP013','VP015','VP016','VP018','VP019'};

load('D:\Project\2021\NIRS\20210203\Result\Ch_order.mat');

for cond = 1:123
    for sub = 1:15
        disp(sub)
        load(['D:\Project\2021\NIRS\20210203\Result\Selected_' num2str(sub) '.mat'])
        for fold = 1:10
            for feat = 1:15
                S{cond}{sub}{fold}{feat} = length(unique(ch.opnum(1,CH.order{cond}{sub}{fold}(1:feat))));
                D{cond}{sub}{fold}{feat} = length(unique(ch.opnum(2,CH.order{cond}{sub}{fold}(1:feat))));
                coord.Clocx{cond}{sub}{fold}{feat} = ch.cxloc(CH.order{cond}{sub}{fold}(1:feat));
                coord.Clocy{cond}{sub}{fold}{feat} = ch.cyloc(CH.order{cond}{sub}{fold}(1:feat));
                coord.Slocx{cond}{sub}{fold}{feat} = ch.sxloc(CH.order{cond}{sub}{fold}(1:feat));
                coord.Slocy{cond}{sub}{fold}{feat} = ch.syloc(CH.order{cond}{sub}{fold}(1:feat));
                coord.Dlocx{cond}{sub}{fold}{feat} = ch.dxloc(CH.order{cond}{sub}{fold}(1:feat));
                coord.Dlocy{cond}{sub}{fold}{feat} = ch.dyloc(CH.order{cond}{sub}{fold}(1:feat));
                x = [coord.Slocx{cond}{sub}{fold}{feat} coord.Dlocx{cond}{sub}{fold}{feat}]';
                y = [coord.Slocy{cond}{sub}{fold}{feat} coord.Dlocy{cond}{sub}{fold}{feat}]';
                k = boundary(x,y,0.35);
                
                coord.xv{cond}{sub}{fold}{feat} = x(k);
                coord.yv{cond}{sub}{fold}{feat} = y(k);
                
                ODA(cond,sub,feat,fold) = polyarea(coord.xv{cond}{sub}{fold}{feat},coord.yv{cond}{sub}{fold}{feat})/140; %xmin ymin, xmax ymin, xmax ymax, xmin ymax
                clearvars -except coord cond sub fold feat ODA ch op_info op_loc op_s op_d ch CH opx opy Analysis_date
            end
        end
    end
end

for cond = 1:123
    for sub = 1:15
        disp(sub)
        for fold = 1:10
            for feat = 1:15
                S = length(unique(ch.opnum(1,CH.order{cond}{sub}{fold}(1:feat))));
                D = length(unique(ch.opnum(2,CH.order{cond}{sub}{fold}(1:feat))));
                Ratio(cond,sub,feat,fold) = (S+D)/(2*feat);
                clear S D
            end
        end
    end
end

save('D:\Project\2021\NIRS\20210506\Result\ODA.mat','ODA');
save('D:\Project\2021\NIRS\20210506\Result\coordinate.mat','coord');
save('D:\Project\2021\NIRS\20210506\Result\Ratio.mat','Ratio');