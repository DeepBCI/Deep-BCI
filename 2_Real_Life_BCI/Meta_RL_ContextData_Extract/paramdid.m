function [mode2, param_init2] = paramdid(mode, param_init)
mode.param_BoundL = [1 1 0.01 0.01];
mode.param_BoundU = [1 1 0.5 0.2];
for i = 1 : 4
    rand(floor(mod(sum(clock*10),10000)));
    param_init(i) = rand  * (mode.param_BoundU(i) - mode.param_BoundL(i))  + mode.param_BoundL(i);
end
mode.param_BoundL(4) = param_init(4);
mode.param_BoundU(4) = param_init(4);
mode2 = mode;
param_init2 = param_init;
end