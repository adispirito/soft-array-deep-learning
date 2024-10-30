
%% Allocate Sparse Memory
l_time = 658;
num_elem = 256;
l_xarr = 136; 
l_yarr = 137;
l_zarr = 134;
num_nonzeros = 12783042560;
M = sparse([], [], [], l_time*num_elem, l_xarr*l_yarr*l_zarr, num_nonzeros);

%% Load from Disk