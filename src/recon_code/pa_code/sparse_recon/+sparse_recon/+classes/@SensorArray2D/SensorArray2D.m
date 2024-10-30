%Sensor Array object
classdef SensorArray2D
    
    %Sensor Array properties
    properties
        x_trans
        y_trans
        z_trans
        IR1Way
        IR1Way_params
    end

    properties (Dependent)
        num_elem
    end

    methods

        %Constructor - load the Sensor Array config
        function obj = SensorArray2D(params_dir)
            get_sensor_locs_fcn = @sparse_recon.utils.get_sensor_locs_fcn;
            [x_trans, y_trans, z_trans] = get_sensor_locs_fcn(params_dir);

            obj.x_trans  = x_trans;
            obj.y_trans  = y_trans;
            obj.z_trans  = z_trans;
            obj.IR1Way_params = load([params_dir,'\','PA_IR.mat']);
            obj.IR1Way = obj.IR1Way_params.IR1Way;

        end

        function num_elem = get.num_elem(obj)
            num_elem = max(size(obj.x_trans));
        end

        %Convert to GPU arrays if desired
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
            obj.x_trans  = gpuArray(obj.x_trans);
            obj.y_trans  = gpuArray(obj.y_trans);
            obj.z_trans  = gpuArray(obj.z_trans);
            obj.IR1Way  = gpuArray(obj.IR1Way);
            warning('on','all');

        end

        function obj = transpose(obj)
            obj.x_trans  = obj.x_trans';
            obj.y_trans  = obj.y_trans';
            obj.z_trans  = obj.z_trans';
        end
    end
end