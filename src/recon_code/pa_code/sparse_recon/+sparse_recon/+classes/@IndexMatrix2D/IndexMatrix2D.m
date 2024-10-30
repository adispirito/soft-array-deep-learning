classdef IndexMatrix2D
    
    properties
        M
        dists
        elem_sens
    end
    
    %primary constructor
    methods

        function obj = IndexMatrix2D(sensorarr, recongrid, dataset, version, gpu_id)
            
            switch version
                case "matrix"       
                    %Sensor array dimensions
                    lxar        = length(recongrid.x_arr);
                    lx0         = length(sensorarr.x_trans);
                    lyar        = length(recongrid.y_arr);
                    ly0         = length(sensorarr.y_trans);
                    lzar        = length(recongrid.z_arr);
                    lz0         = length(sensorarr.z_trans);
                    
                    %Calculate the delay matrix/indices
                    %(x,y,z,elem)
                    xray(:,1,1,:)         = recongrid.x_arr*ones(1, lx0)-ones(lxar,1)*sensorarr.x_trans;
                    yray(1,:,1,:)         = recongrid.y_arr*ones(1, ly0)-ones(lyar,1)*sensorarr.y_trans;
                    zray(1,1,:,:)         = recongrid.z_arr*ones(1, lz0)-ones(lzar,1)*sensorarr.z_trans;
                    
                    % calculate ind, dataset.c: sos/1e3 (1.54), dataset.dt:
                    % 1/decimSampleRate,MHz/1e6 (20)
                    allrays             = hypot(hypot(repmat(xray,1,lyar,lzar,1), ...
                        repmat(yray,lxar,1,lzar,1)), repmat(zray,lxar,lyar,1,1)).*...
                        (1/dataset.c/dataset.dt);
                    obj.dists = allrays;
                    flrays = round(allrays);
                    
                    %Get the row numbers
                    I                   = flrays > 0 & flrays < size(dataset.rfdata,1);
                    flrays(~I)          = 1;
                    
                    %Rows for what
                    Rows                = cast(repmat((1:lxar*lyar*lzar).',lx0,1),'double');
                    
                    %Get the column numbers from the indices
                    phasor1(1,1,1,:)    = (0:size(dataset.rfdata,2)-1)*(size(dataset.rfdata,1));
                    phasorcol           = repmat(phasor1,lxar,lyar,lzar,1);
                    Cols                = double(flrays + phasorcol);

                    %Generate the sparse DAS matrix
                    obj.M               = sparse(Rows, Cols(:), ones(numel(allrays),1), lyar*lxar*lzar, numel(dataset.rfdata));
                case "fast_matrix"
                    sensorarr = sensorarr.transpose();

                    %Sensor array / Recon grid dimensions
                    l_xarr   = length(recongrid.x_arr);
                    l_yarr   = length(recongrid.y_arr);
                    l_zarr   = length(recongrid.z_arr);

                    num_elem = dataset.num_elements;
                    l_time = size(dataset.rfdata, 1);
                    
                    %Calculate the delay matrix/indices
                    %(x,y,z,elem)
                    dist_x = zeros(l_xarr, 1, 1, num_elem, ...
                                   "double");
                    dist_y = zeros(1, l_yarr, 1, num_elem, ...
                                   "double");
                    dist_z = zeros(1, 1, l_zarr, num_elem, ...
                                   "double");
                    
                    warning('off','all');
                    dist_x(:, 1, 1, :) = pdist2(recongrid.x_arr, sensorarr.x_trans, ...
                                                "fasteuclidean", CacheSize="maximal");
                    dist_y(1, :, 1, :) = pdist2(recongrid.y_arr, sensorarr.y_trans, ...
                                                "fasteuclidean", CacheSize="maximal");
                    dist_z(1, 1, :, :) = pdist2(recongrid.z_arr, sensorarr.z_trans, ...
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
                    xyz_rays = xyz_rays .* (1 / (dataset.c) / dataset.dt);

                    obj.dists = xyz_rays;
                    xyz_rays = round(xyz_rays);
                    
                    %Get the row numbers
                    I            = xyz_rays > 0 & xyz_rays < l_time;
                    xyz_rays(~I) = 1;
                    
                    %Rows for what
                    Rows = repmat((1 : l_xarr*l_yarr*l_zarr).', num_elem, 1);
                    Rows = double(Rows);
                    
                    %Get the column numbers from the indices
                    phasorcol(1, 1, 1, :) = (0 : num_elem-1)*(l_time);
                    phasorcol             = repmat(phasorcol, l_xarr, ...
                                                   l_yarr, l_zarr, 1);
                    Cols = xyz_rays + phasorcol;
                    num_rays = numel(xyz_rays);
                    clear phasorcol xyz_rays
                    Cols = double(Cols);

                    %Generate the sparse DAS matrix
                    obj.M = sparse(Rows, Cols(:), ones(num_rays, 1), ...
                                   l_yarr*l_xarr*l_zarr, numel(dataset.rfdata(:, :, 1)));
                case "fast_matrix_gpu"
                    sensorarr = sensorarr.transpose();
                    sensorarr = sensorarr.to_gpu(gpu_id);
                    recongrid = recongrid.to_gpu(gpu_id);
                    

                    %Sensor array / Recon grid dimensions
                    l_xarr   = length(recongrid.x_arr);
                    l_yarr   = length(recongrid.y_arr);
                    l_zarr   = length(recongrid.z_arr);

                    num_elem = dataset.num_elements;
                    l_time = size(dataset.rfdata, 1);
                    
                    %Calculate the delay matrix/indices
                    %(x,y,z,elem)
                    warning('off','all');
                    dist_x(:, 1, 1, :) = pdist2(recongrid.x_arr, sensorarr.x_trans);
                    dist_y(1, :, 1, :) = pdist2(recongrid.y_arr, sensorarr.y_trans);
                    warning('on','all');

                    % calculate ind, dataset.c: sos/1e3 (1.54), dataset.dt:
                    % 1/decimSampleRate,MHz/1e6 (20)
                    dist_x = repmat(gather(dist_x), 1, l_yarr, l_zarr, 1);
                    dist_y = repmat(gather(dist_y), l_xarr, 1, l_zarr, 1);
                    xy_rays = hypot(gpuArray(dist_x), gpuArray(dist_y));
                    xy_rays = gather(xy_rays);
                    [dist_x, dist_y] = deal(0); clear dist_x dist_y; %#ok<ASGLU>
                    warning('off','all');
                    dist_z(1, 1, :, :) = pdist2(recongrid.z_arr, sensorarr.z_trans);
                    warning('on','all');
                    dist_z = repmat(gather(dist_z), l_xarr, l_yarr, 1, 1);
                    xyz_rays = hypot(gpuArray(xy_rays), gpuArray(dist_z));
                    [dist_z, xy_rays] = deal(0); clear dist_z xy_rays; %#ok<ASGLU>
                    xyz_rays = gather(xyz_rays);
                    
                    %ang_sens_map = obj.calc_ang_sens_map_gpu(recongrid, sensorarr, dataset, xyz_rays);

                    xyz_rays = xyz_rays / 1000; % Convert mm to m
                    % dataset.c is in m/s and dataset.dt is in s
                    xyz_rays = xyz_rays .* (1 / dataset.c / dataset.dt);
                    obj.dists = xyz_rays;
                    xyz_rays = round(gpuArray(xyz_rays));
                    xyz_rays = gather(xyz_rays);
                    %Get the row numbers
                    I            = xyz_rays > 0 & xyz_rays < l_time;
                    xyz_rays(~I) = 1;
                    clear I;
                    
                    %Rows for what
                    Rows = repmat((1 : l_xarr*l_yarr*l_zarr).', num_elem, 1);
                    % Rows = double(Rows);
                    Rows = uint32(Rows);
                    
                    %Get the column numbers from the indices
                    phasorcol(1, 1, 1, :) = (0 : (num_elem-1))*(l_time);
                    phasorcol             = repmat(phasorcol, l_xarr, ...
                                                   l_yarr, l_zarr, 1);
                    Cols = xyz_rays + phasorcol;
                    num_rays = numel(xyz_rays);
                    [phasorcol, xyz_rays] = deal(0); clear phasorcol xyz_rays; %#ok<ASGLU>
                    %Cols = double(Cols);
                    Cols = uint32(Cols);

                    %Generate the sparse DAS matrix (in CSC format)
                    obj.M = sparse(Rows, Cols(:), ones(num_rays, 1, "double"), ...
                                   l_yarr*l_xarr*l_zarr, numel(dataset.rfdata(:, :, 1)));
                    
                    clear Rows Cols;
                    obj = obj.to_gpu(gpu_id);
            end
        end
        
        function obj = to_gpu(obj, gpu_id)
            arguments
                obj
                gpu_id {mustBeNumeric} = 0
            end
            if gpu_id == 0
                % Do Nothing
            elseif gpu_id == -1
                [~] = gpuDevice();
            else
                [~] = gpuDevice(gpu_id);
            end
            warning('off','all');
            obj.M = gpuArray(obj.M);
            warning('on','all');
        end
    end
    methods(Static)
        function ang_sens_map = calc_ang_sens_map_gpu(recongrid, sensorarr, dataset, dists)
            %Sensor array / Recon grid dimensions
            l_xarr   = length(recongrid.x_arr);
            l_yarr   = length(recongrid.y_arr);
            l_zarr   = length(recongrid.z_arr);

            num_elem = dataset.num_elements;
            l_time = size(dataset.rfdata, 1);
            ang_sens_map = zeros(l_xarr, l_yarr, l_zarr, num_elem, 'single');
            % [xsize,ysize,zsize] = size(x_img0);
            % x_img1 = reshape(x_img0,[xsize*ysize*zsize,1]);
            % y_img1 = reshape(y_img0,[xsize*ysize*zsize,1]);
            % z_img1 = reshape(z_img0,[xsize*ysize*zsize,1]);

            
            % for ielem = 1:num_elements
            %     dotUV = x_trans(ielem)*(x_img1 - x_trans(ielem)) + ...
            %         y_trans(ielem)*(y_img1 - y_trans(ielem)) + ...
            %         z_trans(ielem)*(z_img1 - z_trans(ielem));
            %     normU = norm([x_trans(ielem) y_trans(ielem) z_trans(ielem)]);
            %     normV = sqrt((x_img1 - x_trans(ielem)).^2 + ...
            %         (y_img1 - y_trans(ielem)).^2 + ...
            %         (z_img1 - z_trans(ielem)).^2);
            %     temp = abs(dotUV./(normU*normV));
            %     temp(temp>1) = 1;
            %     %     temp(temp<-1) = -1;
            %     theta0 = acos(temp);
            %     angleAll(:,:,:,ielem) = reshape(theta0,[xsize,ysize,zsize]);
            % end
            
            for ielem = 1:num_elements
                dotUV = x_trans(ielem)*(x_img1 - x_trans(ielem)) + ...
                    y_trans(ielem)*(y_img1 - y_trans(ielem)) + ...
                    z_trans(ielem)*(z_img1 - z_trans(ielem));
                normU = norm([x_trans(ielem) y_trans(ielem) z_trans(ielem)]);
                normV = sqrt((x_img1 - x_trans(ielem)).^2 + ...
                    (y_img1 - y_trans(ielem)).^2 + ...
                    (z_img1 - z_trans(ielem)).^2);
                temp = abs(dotUV./(normU*normV));
                temp(temp>1) = 1;
                %     temp(temp<-1) = -1;
                theta0 = acos(temp);
                angleAll(:,:,:,ielem) = reshape(theta0,[xsize,ysize,zsize]);
            end
        end
    end
end