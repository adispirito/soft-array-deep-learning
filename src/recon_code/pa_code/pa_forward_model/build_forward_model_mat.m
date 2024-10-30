clear all;
clc;
KernelGenCls = @KernelGenCls;
cd ../
cd sparse_recon%soft-array-deep-learning/src/recon_code/pa_code/sparse_recon
addpath('../sparse_recon')
SensorArray2D = @sparse_recon.classes.SensorArray2D;
ReconGrid2D = @sparse_recon.classes.ReconGrid2D;
%% Loading/Config Impulse Response:
params_dir = './+sparse_recon/params/';
imp_resp_params = load([params_dir, 'PA_IR.mat']);
kernel = imp_resp_params.IR1Way;
hilb_k = abs(hilbert(kernel));
k_max_ind = find(hilb_k == max(hilb_k));
k_adj = length(kernel) - 2*k_max_ind;
% kernel = kernel - mean(kernel);
% kernel = kernel / max(abs(kernel));
kernel = kernel / max(kernel);
kernel = padarray(kernel, k_adj, 'pre');

%% Set Sensor Array
sensorarr = SensorArray2D(params_dir);

%% Set Recon Grid
x_range = [-2.0, 2];%[-4.1, 4];
y_range = [-2.0, 2];%[-4, 4.2];
z_range = [-2.0, 2];%[-4, 4];
Res = 0.06; % in mm

x_area.min = x_range(1);
x_area.max = x_range(2);
x_area.ds = Res;
y_area.min = y_range(1);
y_area.max = y_range(2);
y_area.ds = Res;
z_area.min = z_range(1);
z_area.max = z_range(2);
z_area.ds = Res;

recongrid = ReconGrid2D(x_area, y_area, z_area);

cd ../
cd ./pa_forward_model
%% Calc Params
%%{
temp_c = imp_resp_params.temperature_C;
sos = (1402.4+5.01*temp_c-0.055*temp_c^2+0.00022*temp_c^3);
Receive = imp_resp_params.Receive;
fs = Receive(1).decimSampleRate*1e6; % in Hz (1/s)
target_fs = fs/4;%fs / 4;
dt = 1/target_fs;
%target_temp = temp_c;
target_temp = 24; % degrees C
target_sos = (1402.4+5.01*target_temp-0.055*target_temp^2+0.00022*target_temp^3);

%Further sparsify kernel manually
kernel(1:38) = 0;

%Interpolate to Target Fs
kernel = imresize(kernel, 'Scale', [target_fs/fs, 1], 'Method', 'lanczos3');
kernel = kernel / max(kernel, [], "all");
kernel(abs(kernel) < 0.05) = 0;
% kernel(26:end) = 0;

%Interpolate/Pad to new SOS
orig_length = length(kernel);
kernel = imresize(kernel, 'Scale', [target_sos/sos, 1], 'Method', 'lanczos3');
kernel = kernel / max(kernel, [], "all");
kernel(abs(kernel) < 0.05) = 0;
if length(kernel) < orig_length
    before = floor((orig_length - length(kernel))/2);
    after = ceil((orig_length - length(kernel))/2);
    kernel = padarray(kernel, before, 'pre');
    kernel = padarray(kernel, after, 'post');
elseif length(kernel) > orig_length
    before = floor((length(kernel) - orig_length)/2);
    after = ceil((length(kernel) - orig_length)/2);
    kernel = kernel(before+1:end-after);
end
sos = target_sos;

%Further sparsify kernel manually
%kernel([1:10, 22:end]) = 0;

%Sensor array / Recon grid dimensions
l_xarr   = length(recongrid.x_arr);
l_yarr   = length(recongrid.y_arr);
l_zarr   = length(recongrid.z_arr);

num_elem = sensorarr.num_elem;

%Calculate the delay matrix/indices
%(x,y,z,elem)
dist_x = zeros(l_xarr, 1, 1, num_elem, ...
               "double");
dist_y = zeros(1, l_yarr, 1, num_elem, ...
               "double");
dist_z = zeros(1, 1, l_zarr, num_elem, ...
               "double");

warning('off','all');
dist_x(:, 1, 1, :) = pdist2(recongrid.x_arr, sensorarr.x_trans', ...
                            "fasteuclidean", CacheSize="maximal");
dist_y(1, :, 1, :) = pdist2(recongrid.y_arr, sensorarr.y_trans', ...
                            "fasteuclidean", CacheSize="maximal");
dist_z(1, 1, :, :) = pdist2(recongrid.z_arr, sensorarr.z_trans', ...
                            "fasteuclidean", CacheSize="maximal");
warning('on','all');

% calculate ind, dataset.c: sos/1e3 (1.54), dataset.dt:
% 1/decimSampleRate,MHz/1e6 (20)
dist_x = repmat(dist_x, 1, l_yarr, l_zarr, 1);
dist_y = repmat(dist_y, l_xarr, 1, l_zarr, 1);
xy_rays = hypot(dist_x, dist_y);
clear dist_x dist_y
dist_z = repmat(dist_z, l_xarr, l_yarr, 1, 1);
xyz_rays = hypot(xy_rays, dist_z);
xyz_rays = xyz_rays / 1000; % Convert from mm to m
clear dist_z xy_rays
% dataset.c is in m/s and dataset.dt is in s
t_xyz_rays = xyz_rays .* (1 / (sos) / dt);
max_t_ind = round(max(t_xyz_rays, [], 'all'));
min_t_ind = round(min(t_xyz_rays, [], 'all'));
%}

%% Forward Model Params:
l_time = max_t_ind; % can set as custom (num rows of real channel data)
loc_of_imp_resp = [0, 0, 0] ; % in mm in relation to the focus as origin
loc_of_imp_resp = loc_of_imp_resp; % convert from mm to m
acq_elem_id = 1;
% KernelGenerator = KernelGenCls(kernel, recongrid, sensorarr, xyz_rays, ...
%                                acq_elem_id, loc_of_imp_resp, Res, sos, dt, l_time);
KernelGenerator = KernelGenCls(kernel, recongrid, sensorarr, xyz_rays, ...
                               acq_elem_id, loc_of_imp_resp, Res, target_sos, dt, l_time);
%clear t_xyz_rays xyz_rays recongrid sensorarr
l_time = KernelGenerator.l_time;

tot_grid_points = l_xarr*l_yarr*l_zarr;
max_kernel_non_zero = sum(kernel ~= 0, "all");
rows = zeros(max_kernel_non_zero*num_elem*tot_grid_points, 1, 'uint32');
cols = zeros(max_kernel_non_zero*num_elem*tot_grid_points, 1, 'uint32');
vals = zeros(max_kernel_non_zero*num_elem*tot_grid_points, 1, 'double');
times = [];
iters = 1;
num_nonzeros = 0;
trend = 0;
wb = waitbar(0,'Calculating forward matrix...');
fprintf("\nCalculating forward matrix...\n")
tic
%{
update_freq = 50000;
for i = 1:tot_grid_points
    arr = KernelGenerator.calc_adj_kernels(i);
    [row, col, v] = find(arr(:));
    num_curr_nonzero = length(row);
    if num_curr_nonzero > 0
        rows(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = uint32(row);
        cols(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = uint32(i);
        vals(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = double(v);
    end
    num_nonzeros = num_nonzeros + num_curr_nonzero;
    if mod(i, update_freq) == 0 || i == 1 || i == tot_grid_points
        times(iters) = toc;
        fprintf("Done with %d out of %d in %f sec...\n", i, tot_grid_points, times(iters))
        if mod(i, update_freq) == 0
            travg = times(iters)/update_freq;
            trrem = (tot_grid_points - i) * travg;
            waitbar(i/tot_grid_points, wb, ...
                    [sprintf('%12.1f', trrem/60) ' min remaining for sparse mat construction']);
        elseif i == 1
            travg = times(iters);
            trrem = (tot_grid_points - i) * travg;
            waitbar(i/tot_grid_points, wb, ...
                    [sprintf('%12.1f', trrem/60) ' min remaining for sparse mat construction']);
        end
        iters = iters + 1;
        tic
    end
end
%}

update_freq = 8;
for i = 1:num_elem
    arr = KernelGenerator.calc_adj_kernels(i);
    [row, col, v] = find(arr);
    num_curr_nonzero = length(row);
    if num_curr_nonzero > 0
        rows(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = uint32(row + (i-1)*l_time);
        cols(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = uint32(col);
        vals(num_nonzeros+1:num_curr_nonzero+num_nonzeros) = double(v);
    end
    num_nonzeros = num_nonzeros + num_curr_nonzero;
    if mod(i, update_freq) == 0 || i == 1 || i == num_elem
        times(iters) = toc;
        fprintf("Done with %d out of %d in %f sec...\n", i, num_elem, times(iters))
        if mod(i, update_freq) == 0
            % travg = times(iters)/update_freq;
            travg = mean(times(2:iters))/update_freq;
            trrem = (num_elem - i) * travg;
        elseif i == 1
            travg = times(iters);
            trrem = (num_elem - i) * travg;
        end
        waitbar((i+1)/num_elem, wb, ...
                [sprintf('%12.1f', trrem/60) ' min remaining for sparse mat construction']);
        iters = iters + 1;
        tic
    end
end

close(wb)
fprintf("Done!!!\n")

if num_nonzeros < length(vals)
    fprintf("Indexing just nonzero vals...\n")
    rows = rows(1:num_nonzeros);
    cols = cols(1:num_nonzeros);
    vals = vals(1:num_nonzeros);
    fprintf("Done!!!\n")
end

% rows = [];
% cols = [];
% vals = [];
% tot_grid_points = l_xarr*l_yarr*l_zarr;
% update_freq = 1000;
% fprintf("\nCalculating forward matrix...\n")
% for i = 1:tot_grid_points
%     tic
%     arr = KernelGenerator.calc_adj_kernels(i);
%     [row, col, v] = find(arr > 0);
%     rows = [rows, row];
%     cols = [cols, col*i];
%     vals = [vals, v];
% 
%     if mod(i, update_freq) == 0 || i == 1
%         fprintf("Done with %d out of %d...\n", i, l_xarr*l_yarr*l_zarr)
%         toc
%     end
% end
%%{
%clearvars -except rows cols vals l_time num_elem l_xarr l_yarr l_zarr
tic
fprintf("\nFilling Sparse Matrix...\n")
forward_mat = sparse(rows, cols, vals, l_time*num_elem, l_xarr*l_yarr*l_zarr);
% forward_mat = sparse(rows(1:num_nonzeros), ...
%                      cols(1:num_nonzeros), ...
%                      vals(1:num_nonzeros), ...
%                      l_time*num_elem, ...
%                      l_xarr*l_yarr*l_zarr);
fprintf("Done!!!\n")
toc
%}