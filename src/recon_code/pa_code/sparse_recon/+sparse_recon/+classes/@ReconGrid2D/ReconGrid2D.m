%Sensor Configuration

classdef ReconGrid2D

    properties
        x_arr
        y_arr
        z_arr
        grid_size
    end

    methods
    
        %Constructor
        function obj = ReconGrid2D(x_area, y_area, z_area)
            xmax = x_area.max;
            xmin = x_area.min;
            dx = x_area.ds;
            
            ymax = y_area.max;
            ymin = y_area.min;
            dy = y_area.ds;
            
            zmax = z_area.max;
            zmin = z_area.min;
            dz = z_area.ds;
            
            obj.x_arr = (xmin:dx:xmax).';
            obj.y_arr = (ymin:dy:ymax).';
            obj.z_arr = (zmin:dz:zmax).';

            obj.grid_size = [length(obj.x_arr), length(obj.y_arr), length(obj.z_arr)];
    
        end
    
        %GPU version of imaging area object.
        function obj = to_gpu(obj, gpu_id)
            arguments
                obj
                gpu_id {mustBeNumeric} = 0
            end
            warning('off','all');
            if gpu_id == 0
                % Do Nothing
            elseif gpu_id == -1
                [~] = gpuDevice();
            else
                [~] = gpuDevice(gpu_id);
            end
            obj.x_arr = gpuArray(single(obj.x_arr));
            obj.y_arr = gpuArray(single(obj.y_arr));
            obj.z_arr = gpuArray(single(obj.z_arr));
            warning('on','all');
        end
    end
end