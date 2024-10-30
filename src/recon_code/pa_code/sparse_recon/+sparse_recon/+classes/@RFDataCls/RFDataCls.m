% Dataset Class

classdef RFDataCls

    %dataset properties
    properties
        fs
        c
        rfdata
        elem_width
        elem_widthwl
    end
    properties (Dependent)
        num_elements
        l_time
        num_frames
        dt
    end
    
    methods
        
        %Constructor - load the dataset
        function obj = RFDataCls(rfdata, fs, sos, trans)
            
            obj.fs           = fs;
            % obj.dt           = 1/(fs);
            obj.c            = sos;
            obj.rfdata       = rfdata;
            obj.elem_width   = trans.elementWidth; % mm
            obj.elem_widthwl = obj.elem_width/(sos/1e3/trans.frequency);
            % obj.num_elements = trans.numelements;
            % obj.l_time = size(rfdata, 1);
            % obj.num_frames = size(rfdata, 3);
    
        end
        
        % Ensure that when rfdata changes, so does l_time
        function l_time = get.l_time(obj)
            l_time = size(obj.rfdata, 1);
        end
        % function obj = set.l_time(obj, l_time)
        %     obj.l_time = l_time;
        % end

        % Ensure that when rfdata changes, so does num_elements
        function num_elements = get.num_elements(obj)
            num_elements = size(obj.rfdata, 2);
        end

        % Ensure that when rfdata changes, so does num_frames
        function num_frames = get.num_frames(obj)
            num_frames = size(obj.rfdata, 3);
        end

        % Ensure that when fs changes, so does dt
        function dt = get.dt(obj)
            dt = 1/(obj.fs);
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
            obj.rfdata = gpuArray(obj.rfdata);
            warning('on','all');
        end
    
        %Change the rfdata data type (in case you need to for sparse mmult)
        function obj = rfcaster(obj, newclass)
            obj.rfdata = cast(obj.rfdata, newclass);
        end

    end

end