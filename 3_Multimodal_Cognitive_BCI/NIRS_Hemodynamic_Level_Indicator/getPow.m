function [trialNum, getPow1, getPow2] = getPow(pack)

    pack = strtrim(pack);
    
    cutindex1 = strfind(pack,'<');
    cutindex2 = strfind(pack,'>');
    cutindex3 = strfind(pack,',');
    
    
    for i = (cutindex1 + 1) : (cutindex3(1) - 1)
        trial(i - cutindex1 ) = pack(i);
    end
    
    
    for i = (cutindex3(1) + 1) : (cutindex3(2) - 1)
        pow1(i - (cutindex3(1)) ) = pack(i);
    end
    
    
    for i = (cutindex3(2) + 1) : (cutindex2 - 1)
        pow2(i - (cutindex3(2)) ) = pack(i);
    end
    
    
    trial = strtrim(trial);
    pow1 = strtrim(pow1);
    pow2 = strtrim(pow2);
    
    trialNum = str2num(trial);
    getPow1 = str2num(pow1);
    getPow2 = str2num(pow2);

end