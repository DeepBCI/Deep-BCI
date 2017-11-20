function str = secs2hmsstr(secs)

days = floor(secs/(3600*24));
rem = (secs-(days*3600*24));
hours = floor(rem/3600);
rem = rem -(hours*3600);
minutes = floor(rem/60);
rem = rem - minutes*60;
secs = round(rem);

switch days
    case 0
        str = [ num2str(hours) ':' sprintf('%02d',minutes) ':' sprintf('%02d',secs)];
    case 1
        str = [ '1 Day + ' num2str(hours) ':' sprintf('%02d',minutes) ':' sprintf('%02d',secs)];
    otherwise
        str = [ num2str(days) ' Days + ' num2str(hours) ':' sprintf('%02d',minutes) ':' sprintf('%02d',secs)];
end