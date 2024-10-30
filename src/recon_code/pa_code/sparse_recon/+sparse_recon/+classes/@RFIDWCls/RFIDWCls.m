classdef RFIDWCls
    %RFINTERPOLANTCLS Summary of this class goes here
    
    properties
        trans_locs
        data
        l_time
        num_frames
        k_min
        k_max
        radius
        exp_factor
        decor_dist
    end
    
    methods
        function obj = RFIDWCls(trans_locs, data_inst, radius, k_min, k_max, ...
                                exp_factor, decor_dist)
            %RFINTERPOLANTCLS Construct an instance of this class
            arguments
                trans_locs
                data_inst
                radius = 5 % mm
                k_min = 1%3;
                k_max = 3
                exp_factor = 2%2
                decor_dist = 1 %0.150 % mm
            end
            % delete(gcp('nocreate'))
            % parpool("threads");
            obj.trans_locs = trans_locs;
            obj.data = data_inst.rfdata;
            obj.l_time = data_inst.l_time;
            obj.num_frames = data_inst.num_frames;
            obj.radius = radius;
            obj.k_min = k_min;
            obj.k_max = k_max;
            obj.exp_factor = exp_factor;
            obj.decor_dist = decor_dist;
        end

        function interp_data = interp(obj, pos_q, method)
            arguments
                obj
                pos_q
                method = "IDW"%"IDW"
            end
            dists = pdist2(pos_q', obj.trans_locs', ...
                           "fasteuclidean", CacheSize="maximal");
            [min_vals, min_inds] = mink(dists, obj.k_max, 2);
            within_rad = min_vals <= obj.radius;
            num_valid_nearest = sum(within_rad, 2);
            where_below_kmin = num_valid_nearest < obj.k_min;
            w_inds_excl = zeros(size(min_vals));
            w_inds_excl(~within_rad) = 1;
            w_inds_excl(where_below_kmin, 1:obj.k_min) = 0;
            w_inds_excl = logical(w_inds_excl);
            clear dists within_rad num_valid_nearest where_below_kmin;
            interp_data = zeros(obj.l_time, length(pos_q), obj.num_frames);
            rfdata = obj.data;
            for i = 1:length(pos_q)
                k_dists = min_vals(i, :);
                k_inds = min_inds(i, :);
                k_rfdata = rfdata(:, k_inds, :);
                w = obj.weight_method(k_dists, method);
                w(w_inds_excl(i, :)) = 0;
                if sum(w, "all") == 0
                    interp_data(:, i, :) = 0;
                else
                    % interp_data(:, i, :) = sum(k_rfdata .* w, 2) ./ sum(w, "all");
                    % interp_data(:, i, :) = (1/(min(k_dists, [], "all") + 1)) .* (sum(k_rfdata .* w, 2) ./ sum(w, "all"));
                    interp_dist_decay = (1/(min(k_dists, [], "all")/2 + 1));
                    interp_data(:, i, :) = interp_dist_decay .* (sum(k_rfdata .* w, 2) ./ sum(w, "all"));
                end
            end
        end
        
        function weights  = weight_method(obj, k_dists, method)
            switch method
                case "IDW"
                    % weights = 1 ./ (k_dists .^ obj.exp_factor);
                    weights = 1 ./ ((k_dists .^ obj.exp_factor) + 1);
                case "custom_IDW"
                    % weights = 1 ./ (k_dists .^ obj.exp_factor);
                    weights = 1 ./ (((k_dists / (obj.radius/2)).^ obj.exp_factor) + 1);
                case "exp"
                    weights = exp(-(k_dists ./ obj.decor_dist) .^ obj.exp_factor);
                case "linear"
                    weights = obj.radius - k_dists;
            end
        end
    end
end

