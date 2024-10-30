classdef RFInterpolantCls
    %RFINTERPOLANTCLS Summary of this class goes here
    
    properties
        F
        l_time
        num_frames
    end
    
    methods
        function obj = RFInterpolantCls(x, y, z, data)
            %RFINTERPOLANTCLS Construct an instance of this class
            delete(gcp('nocreate'))
            parpool("threads");
            %%{
            CORR_THRES = 0.9;
            shape = size(data);
            corrs = [];
            tic
            for i = 1:shape(end)
                data1 = data(:, :, 1);
                datai = data(:, :, i);
                corre = corrcoef(data1(:), datai(:));
                corrs(i) = corre(1, 2);
            end
            toc
            clean_frames = corrs >= CORR_THRES;
            clean_data = data(:, :, clean_frames);
            avg_data = squeeze(mean(clean_data, 3));
            clear clean_data
            %}
            shape = size(data);
            avg_data = squeeze(data(:, :, 1));
            l_time = shape(1);
            F = cell(l_time, 1);
            tic
            parfor i = 1:l_time
                data = squeeze(avg_data(i, :))';
                F{i} = scatteredInterpolant(x', y', z', ...
                                            data, 'natural', 'linear');
            end
            toc
            obj.F = F;
            obj.l_time = l_time;
            obj.num_frames = shape(end);
        end

        function interp_data = interp(obj, xq, yq, zq)
            assert((length(xq)==length(yq)) && (length(xq)==length(zq)))
            interp_data = zeros(obj.l_time, length(xq), obj.num_frames);
            func = obj.F;
            number_frames = obj.num_frames;
            tic
            parfor t = 1:obj.l_time
                interp_data(t, :, :) = repmat(func{t}(xq, yq, zq)', 1, number_frames);
                %interp_data(t, :, 1) = func{t}(xq, yq, zq)';
            end
            toc
        end
       
    end
end

