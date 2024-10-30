%Kernel Generator object
classdef KernelGenCls
    
    properties
        sos
        dt
        l_time
        k_adj
        acq_elem_id
        dist_elem_to_imp_resp
        sensorarr
        recongrid
        dists
    end

    methods

        %Constructor - load the kernel generator config
        function obj = KernelGenCls(kernel, recongrid, sensorarr, dists, ...
                                    acq_elem_id, loc_of_imp_resp, ...
                                    Res, sos, dt, l_time)
            delete(gcp('nocreate'))
            parpool('Threads');
            obj.acq_elem_id = acq_elem_id;
            obj.sos = sos;
            obj.dt = dt;
            obj.l_time = l_time;
            obj.sensorarr = sensorarr;
            obj.dists = dists;
            obj.recongrid = recongrid;
            [x_foc_pos, x_foc_idx] = min(abs(recongrid.x_arr)); % in mm
            [y_foc_pos, y_foc_idx] = min(abs(recongrid.y_arr)); % in mm
            [z_foc_pos, z_foc_idx] = min(abs(recongrid.z_arr)); % in mm
            imp_resp_peak_inds = [round(x_foc_idx + loc_of_imp_resp(1)/Res), ...
                                  round(y_foc_idx + loc_of_imp_resp(2)/Res), ...
                                  round(z_foc_idx + loc_of_imp_resp(3)/Res)];
            grid_locs_of_imp_resp = [recongrid.x_arr(imp_resp_peak_inds(1)), ...
                                     recongrid.y_arr(imp_resp_peak_inds(2)), ...
                                     recongrid.z_arr(imp_resp_peak_inds(3))]; % in mm
            
            dist_elem_to_imp_resp = dists(imp_resp_peak_inds(1), ...
                                          imp_resp_peak_inds(2), ...
                                          imp_resp_peak_inds(3), ...
                                          acq_elem_id); % in m
            obj.dist_elem_to_imp_resp = dist_elem_to_imp_resp;

            % hilb_k = abs(hilbert(kernel));
            % k_max_ind = find(hilb_k == max(hilb_k));
            % start_to_peak_dist = k_max_ind*dt*sos;
            % elem_to_start_imp_dist = dist_elem_to_imp_resp - start_to_peak_dist;
            % z_dist = sensorarr.z_trans(elem_id);
            % elem_offset = 

            % Pad Kernel to account for min distance between transducer and recon grid
            t_dists = dists .* (1 / (sos) / dt);
            min_t_ind = round(min(t_dists, [], 'all'));
            pad_imp_resp = padarray(kernel, ...
                                    min_t_ind, ...
                                    0, 'pre');
            pad_imp_resp = padarray(pad_imp_resp, ...
                                    l_time - length(pad_imp_resp), ...
                                    0, 'post');
            obj.k_adj = pad_imp_resp;
        end
        
        % function out_kernels = calc_adj_kernels(obj, grid_id)
        %     num_elem = obj.sensorarr.num_elem;
        %     grid_size = obj.recongrid.grid_size;
        %     dists_vec = reshape(obj.dists, [prod(grid_size), num_elem]);
        %     dists_vec = dists_vec(grid_id, :);
        %     out_kernels = zeros(obj.l_time, num_elem);
        %     prime_kernel = obj.k_adj;
        %     prime_dist = obj.dist_elem_to_imp_resp;
        %     conv_factor = (1 / (obj.sos) / obj.dt);
        %     for i = 1:num_elem
        %         adj_dist = dists_vec(i) - prime_dist;
        %         shift = round(adj_dist*conv_factor);
        %         out = circshift(prime_kernel, shift);
        %         if shift > 0
        %             out(1:shift) = 0;
        %         else
        %             out(end+shift+1:end) = 0;
        %         end
        %         out_kernels(:, i) = out;
        %     end
        % end

        function out_kernels = calc_adj_kernels(obj, elem_id)
            num_elem = obj.sensorarr.num_elem;
            grid_size = obj.recongrid.grid_size;
            dists_vec = reshape(obj.dists, [prod(grid_size), num_elem]);
            dists_vec = dists_vec(:, elem_id);
            out_kernels = zeros(obj.l_time, prod(grid_size));
            prime_kernel = obj.k_adj;
            prime_dist = obj.dist_elem_to_imp_resp;
            conv_factor = (1 / (obj.sos) / obj.dt);
            parfor i = 1:prod(grid_size)
                adj_dist = dists_vec(i) - prime_dist;
                shift = round(adj_dist*conv_factor);
                out = circshift(prime_kernel, shift);
                if shift > 0
                    out(1:shift) = 0;
                else
                    out(end+shift+1:end) = 0;
                end
                out_kernels(:, i) = out;
            end
        end
    end
end