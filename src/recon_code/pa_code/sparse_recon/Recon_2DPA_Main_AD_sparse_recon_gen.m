%% Import Packages:
SensorArray2D = @sparse_recon.classes.SensorArray2D;
ReconGrid2D = @sparse_recon.classes.ReconGrid2D;
RFDataCls = @sparse_recon.classes.RFDataCls;
IndexMatrix2D = @sparse_recon.classes.IndexMatrix2D;
RFInterpolantCls = @sparse_recon.classes.RFInterpolantCls;
RFIDWCls = @sparse_recon.classes.RFIDWCls;

%% Set GPU
if recon_params.gpu_id == 0
    % Do Nothing
elseif recon_params.gpu_id == -1
    [~] = gpuDevice();
else
    [~] = gpuDevice(recon_params.gpu_id);
end
GPU_ID = 0;
%% Set Sensor Array
SensorArray2D = SensorArray2D(params_dir);
ranges = {1:8, 9:18, 19:31, 32:46, 47:63, 64:82, 83:104, 105:128, ...
          129:135, 136:145, 146:157, 158:172, 173:190, 191:210, 211:232, 233:256};
%% Set Recon Grid
x_area.min = x_range(1);
x_area.max = x_range(2);
x_area.ds = Res;
y_area.min = y_range(1);
y_area.max = y_range(2);
y_area.ds = Res;
z_area.min = z_range(1);
z_area.max = z_range(2);
z_area.ds = Res;

ReconGrid2D = ReconGrid2D(x_area, y_area, z_area);

%% Set RFData Params
sos = (1402.4+5.01*temperature_C-0.055*temperature_C^2+0.00022*temperature_C^3);
%sos = sos - (sos/1000)*4;
fs = Receive(1).decimSampleRate*1e6;
RFDataObj = RFDataCls(RFData, fs, sos, Trans);

%% Process RF data

if DECONVOLVE
    RFDataObj.rfdata = deconv_pact_rf(RFDataObj, SensorArray2D);
    % deconshift = 5;
    deconshift = 0;
else
    deconshift = 0;
end

RFDataObj.rfdata = imtranslate(RFDataObj.rfdata, [0 -RF_SHIFT(iwl)-deconshift], ...
                               'FillValues', 0);

switch FILTER
    case 'HP'
        fprintf('HPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(RFDataObj.fs/2),'high');
    case 'LP'
        fprintf('LPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(RFDataObj.fs/2),'low');
    case 'BP'
        fprintf('BPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(RFDataObj.fs/2),'bandpass');
end

if ~isempty(FILTER)
   fprintf('Filtering: ')
   tic
   % filtered_data = filter(b,a,RFDataObj.rfdata,[],1);
   filtered_data = filtfilt(b, a, RFDataObj.rfdata);
   RFDataObj.rfdata = filtered_data;
   %RFDataObj.rfdata = imtranslate(filtered_data,[0 -FILTERRFCOMP],'FillValues',min(RFDataObj.rfdata(:)));
   toc
end

if HILBERTRF
    fprintf('RF Data Hilbert Transform: ')
    tic
    RFDataObj.rfdata = hilbert(RFDataObj.rfdata);
    toc
end

%% Remove Breathing Frames
rfdata = single(RFDataObj.rfdata);
if RM_BREATH
    data_size = size(rfdata);
    corrs = [];
    for i = 1:data_size(end)
        data1 = rfdata(:, :, 1);
        data2 = rfdata(:, :, i);
        corre = corrcoef(data1(:), data2(:));
        corrs(i) = corre(1, 2);
    end
    frames = corrs >= (mean(corrs) - CORR_THRES_SIGMA*std(corrs));
    RFDataObj.rfdata = RFDataObj.rfdata(:, :, frames);
end

%% Average Frames if FLAG
if RECON_AVG
    RFDataObj.rfdata = mean(RFDataObj.rfdata, 3);
end

%% Interpolation
if length(size(RFDataObj.rfdata)) == 3
    fprintf("Array Size (Before Interpolation): (%i, %i, %i)\n", size(RFDataObj.rfdata))
else
    fprintf("Array Size (Before Interpolation): (%i, %i, 1)\n", size(RFDataObj.rfdata))
end

switch INTERP
    case "simple_channels+time"
        RFDataObj.num_elements = RFDataObj.num_elements * INTERP_RATIO;
        x_trans_orig = SensorArray2D.x_trans;
        y_trans_orig = SensorArray2D.y_trans;
        z_trans_orig = SensorArray2D.z_trans;
        x_trans = zeros(1, length(x_trans_orig)*INTERP_RATIO);
        y_trans = zeros(1, length(y_trans_orig)*INTERP_RATIO);
        z_trans = zeros(1, length(z_trans_orig)*INTERP_RATIO);
        [time_size, channels, num_frames] = size(RFDataObj.rfdata);
        rfdata = zeros(time_size, channels*INTERP_RATIO, num_frames);
        for i=1:length(ranges)
            range = ranges{i};
            start_i = ((range(1) - 1) * INTERP_RATIO) + 1;
            end_i = (range(end)*INTERP_RATIO);
            x_trans(start_i:end_i) = imresize(x_trans_orig(range), "Scale", [1, INTERP_RATIO], "Method", 'bilinear');
            y_trans(start_i:end_i) = imresize(y_trans_orig(range), "Scale", [1, INTERP_RATIO], "Method", 'bilinear');
            z_trans(start_i:end_i) = imresize(z_trans_orig(range), "Scale", [1, INTERP_RATIO], "Method", 'bilinear');
            rfdata(:, start_i:end_i, :) = imresize3(RFDataObj.rfdata(:, range, :), "Scale", [1, INTERP_RATIO, 1], "Method", 'linear');
        end
        RFDataObj.rfdata = imresize3(rfdata, "Scale", [TIME_INTERP_RATIO, 1, 1], "Method", 'cubic'); % linear / cubic / lanczos3
        RFDataObj.fs = RFDataObj.fs*TIME_INTERP_RATIO;
        SensorArray2D.x_trans = x_trans;
        SensorArray2D.y_trans = y_trans;
        SensorArray2D.z_trans = z_trans;
        clear x_trans y_trans z_trans x_trans_orig y_trans_orig z_trans_orig
    case "fast_channels+time"
        % Time Interpolation
        RFDataObj.rfdata = imresize3(RFDataObj.rfdata, "Scale", ...
                                     [TIME_INTERP_RATIO, 1, 1], ...
                                     "Method", 'cubic'); % linear / cubic / lanczos3
        RFDataObj.fs = RFDataObj.fs*TIME_INTERP_RATIO;
        
        % Channel Interpolation
        RFIDWObj = RFIDWCls([SensorArray2D.x_trans; ...
                             SensorArray2D.y_trans; ...
                             SensorArray2D.z_trans], ...
                            RFDataObj);
        x_trans_orig = SensorArray2D.x_trans;
        y_trans_orig = SensorArray2D.y_trans;
        z_trans_orig = SensorArray2D.z_trans;
        x_trans = interp(x_trans_orig, INTERP_RATIO);
        y_trans = interp(y_trans_orig, INTERP_RATIO);
        z_trans = upsample(z_trans_orig, INTERP_RATIO);
        for i = 1:length(ranges)
            range = ranges{i};
            start_i = ((range(1) - 1) * INTERP_RATIO) + 1;
            end_i = (range(end)*INTERP_RATIO);
            new_range = start_i:end_i;
            if i == 8 || i == 16
                range_past = ranges{i - 1};
                current_ring_z_pos = z_trans_orig(range);
                past_ring_z_pos = z_trans_orig(range_past);
                new_ring_z_pos = (mean(current_ring_z_pos, "all") + ...
                                  mean(past_ring_z_pos, "all")) / 2;
                mask = z_trans(new_range) == 0;
                z_trans(new_range(mask)) = new_ring_z_pos;
            else
                range_next = ranges{i + 1};
                current_ring_z_pos = z_trans_orig(range);
                next_ring_z_pos = z_trans_orig(range_next);
                new_ring_z_pos = (mean(current_ring_z_pos, "all") + ...
                                  mean(next_ring_z_pos, "all")) / 2;
                new_ring_z_pos = current_ring_z_pos + (current_ring_z_pos - new_ring_z_pos);
                mask = z_trans(new_range) == 0;
                z_trans(new_range(mask)) = new_ring_z_pos;
            end
        end
        orig_inds = 1:INTERP_RATIO:length(x_trans);
        mask = zeros(1, length(x_trans));
        mask(orig_inds) = 1;
        new_val_mask = ~mask;
        SensorArray2D.x_trans = x_trans;
        SensorArray2D.y_trans = y_trans;
        SensorArray2D.z_trans = z_trans;
        clear x_trans y_trans z_trans x_trans_orig y_trans_orig z_trans_orig
        fprintf("Interpolating...\n")
        new_rfdata = zeros([RFDataObj.l_time, ...
                            RFDataObj.num_elements * INTERP_RATIO, ...
                            RFDataObj.num_frames]);
        new_rfdata(:, orig_inds, :) = RFDataObj.rfdata;
        rfdata = RFIDWObj.interp([SensorArray2D.x_trans(new_val_mask); ...
                                  SensorArray2D.y_trans(new_val_mask); ...
                                  SensorArray2D.z_trans(new_val_mask)]);
        new_rfdata(:, new_val_mask, :) = rfdata;
        RFDataObj.rfdata = new_rfdata;
        clear new_rfdata rfdata
        RFDataObj.num_elements = RFDataObj.num_elements * INTERP_RATIO;
    case "time"
        RFDataObj.rfdata = imresize3(RFDataObj.rfdata, "Scale", [TIME_INTERP_RATIO, 1, 1], "Method", 'cubic'); % linear / cubic / lanczos3
        RFDataObj.fs = RFDataObj.fs*TIME_INTERP_RATIO;
end

%% Build Sparse Matrix and Recon:
diginum = 4; % digits to save batchsize/frame number info
if ~recon_params.JUST_SAVE_CHANNEL_DATA
    USE_GPU = recon_params.gpu_id > 0;
    if length(size(RFDataObj.rfdata)) == 3
        fprintf("Array Size (After Interpolation): (%i, %i, %i)\n", size(RFDataObj.rfdata))
    else
        fprintf("Array Size (After Interpolation): (%i, %i, 1)\n", size(RFDataObj.rfdata))
    end
    fprintf('Use GPU = %d\n', USE_GPU)
    fprintf('Creating Sparse Matrix... ')
    if USE_GPU
        tic
        IndexMatrix2D = IndexMatrix2D(SensorArray2D, ReconGrid2D, RFDataObj, "fast_matrix_gpu", GPU_ID);
        toc
        disp(size(IndexMatrix2D.M))
        disp(size(RFDataObj.rfdata))
        out_size = [RFDataObj.l_time*RFDataObj.num_elements, RFDataObj.num_frames];
        RFDataObj.rfdata = reshape(RFDataObj.rfdata, out_size);
        RFDataObj.rfdata = double(RFDataObj.rfdata);
        RFDataObj = RFDataObj.to_gpu(GPU_ID);
    else
        tic
        IndexMatrix2D = IndexMatrix2D(SensorArray2D, ReconGrid2D, RFDataObj, "fast_matrix", GPU_ID);
        toc
        disp(size(IndexMatrix2D.M))
        disp(size(RFDataObj.rfdata))
        out_size = [RFDataObj.l_time*RFDataObj.num_elements, RFDataObj.num_frames];
        RFDataObj.rfdata = reshape(RFDataObj.rfdata, out_size);
    end
    fprintf('Performing Sparse Matrix Matrix Multi... ')
    tic
    pa_rec = IndexMatrix2D.M*RFDataObj.rfdata;
    pa_rec = reshape(pa_rec, [ReconGrid2D.grid_size(:)', RFDataObj.num_frames]);
    toc
    disp("Done!!!")
    
    
    %%% Convert to single
    pa_rec = single(gather(pa_rec));

    if isreal(pa_rec)
        xymap = squeeze(max(mean(pa_rec,4),[],3));
        xzmap = squeeze(max(mean(pa_rec,4),[],1))';
        yzmap = squeeze(max(mean(pa_rec,4),[],2))';
    else
        xymap = squeeze(max(abs(mean(pa_rec,4)),[],3));
        xzmap = squeeze(max(abs(mean(pa_rec,4)),[],1))';
        yzmap = squeeze(max(abs(mean(pa_rec,4)),[],2))';
    end

    if SAVE_DATA
        tic
    %     save_paramname = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
    %         num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_params.mat'];
        save_name = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
            num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_recon'];
        if save_dir(end) ~= '\'
            save_dir = [save_dir, '\'];
        end
        checkMakeDir(save_dir)
        if ABSDATA && HILBERTRF
            pa_rec0 = single(abs(pa_rec));
        elseif HILBERTRF
            pa_rec0 = single(cat(5, real(pa_rec),imag(pa_rec)));
        end
        if SAVE_AVG && ~RECON_AVG
            pa_rec0 = squeeze(mean(pa_rec0, 4));
        end
        save([save_dir, save_name], 'pa_rec0','x_img','y_img',...
            'z_img','temperature_C','x_range','y_range','z_range',...
            'laser_type','flash2Qdelay','sos','HILBERTRF','MULTI_SHIFT', ...
            '-v7.3', '-nocompression');
        toc
    end
elseif recon_params.JUST_SAVE_CHANNEL_DATA
    if SAVE_DATA
        tic
    %     save_paramname = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
    %         num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_params.mat'];
        save_name = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
            num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_rfdata'];
        if save_dir(end) ~= '\'
            save_dir = [save_dir, '\'];
        end
        checkMakeDir(save_dir)
        if ABSDATA && HILBERTRF
            RFDataObj.rfdata = hilbert(RFDataObj.rfdata);
            RFDataObj.rfdata = single(abs(RFDataObj.rfdata));
        elseif HILBERTRF
            RFDataObj.rfdata = hilbert(RFDataObj.rfdata);
            RFDataObj.rfdata = single(cat(5, real(RFDataObj.rfdata),imag(RFDataObj.rfdata)));
        else
            RFDataObj.rfdata = single(RFDataObj.rfdata);
        end
        if SAVE_AVG && ~RECON_AVG
            RFDataObj.rfdata = squeeze(mean(RFDataObj.rfdata, 4));
        end
        rfdata = RFDataObj.rfdata;
        save([save_dir, save_name], 'rfdata', 'currentfolder', ...
            '-v7.3', '-nocompression');
        toc
    end
end
%% Subfunctions
function checkMakeDir(path)
    if exist(path,'dir') 
        fprintf('Saving path already exists.\n')
    else
        mkdir(path);
    end
end

function deconv = deconv_pact_rf(rfdata_obj, sensorarr_obj)
    % fprintf('RF Data Deconvolution (Prefiltering): \n')
    % tic
    %pre_decon_filt_cutoff = [0.05 10]*1e6; % Hz
    pre_decon_filt_cutoff = [rfdata_obj.fs/50, rfdata_obj.fs/2 - rfdata_obj.fs/20]; % Hz
    prefilt_order = 9;
    [b,a]    = butter(prefilt_order,pre_decon_filt_cutoff/(rfdata_obj.fs/2),'bandpass');
    filtered_data = filtfilt(b, a, rfdata_obj.rfdata);
    rfdata_obj.rfdata = filtered_data;
    % toc

    fprintf('RF Data Deconvolution: ')
    tic
    scale = 4;
    kernel = sensorarr_obj.IR1Way;

    
    temp_c = sensorarr_obj.IR1Way_params.temperature_C;
    k_sos = (1402.4+5.01*temp_c-0.055*temp_c^2+0.00022*temp_c^3);
    target_sos = rfdata_obj.c;
    
    %Interpolate/Pad to new SOS
    orig_length = length(kernel);
    kernel = imresize(kernel, 'Scale', [target_sos/k_sos, 1], 'Method', 'lanczos3');
    if length(kernel) < orig_length
        before = floor((orig_length - length(kernel))/2);
        after = ceil((orig_length - length(kernel))/2);
        kernel = padarray(kernel, before, 0, 'pre');
        kernel = padarray(kernel, after, 0, 'post');
    elseif length(kernel) > orig_length
        before = floor((length(kernel) - orig_length)/2);
        after = ceil((length(kernel) - orig_length)/2);
        kernel = kernel(before+1:end-after);
    end

   
    background = rfdata_obj.rfdata(end-50:end, :, :);
    noise_var = var(background(:));
    estimated_nsr = noise_var / var(rfdata_obj.rfdata(:));

    kernel_fs = rfdata_obj.fs*scale;
    % k_filter_cutoff = [0.2]*1e6; % Hz
    k_filter_cutoff = [rfdata_obj.fs/100]; % Hz
    order = 9;
    [b,a] = butter(order,k_filter_cutoff/(kernel_fs/2),'high');
    kernel = filtfilt(b, a, kernel);

    hilb_k = abs(hilbert(kernel));
    k_max_ind = find(hilb_k == max(hilb_k));
    k_adj = length(kernel) - 2*k_max_ind;
    kernel = padarray(kernel, k_adj, 0,'pre');

    %{
    up_rfdata = imresize(rfdata_obj.rfdata, 'Scale', [scale, 1], 'Method', 'lanczos3');
    %kernel = resample(kernel, 1, 4);
    %kernel = imresize(kernel, 'Scale', [1/4, 1], 'Method', 'lanczos3');
    %kernel(1:38) = 0;
    kernel = kernel / sum(kernel);
    %kernel = kernel / max(kernel);
    deconv = deconvwnr(up_rfdata, kernel);
    deconv = imresize(deconv, 'Scale', [1/scale, 1], 'Method', 'lanczos3');
    % deconv = deconvwnr(rfdata_obj.rfdata, kernel);
    %}

    %{
    kernel = imresize(kernel, 'Scale', [1/scale, 1], 'Method', 'lanczos3');
    kernel = kernel / sum(kernel);
    deconv = deconvwnr(rfdata_obj.rfdata, kernel);
    %}

    %{
    kernel(1:38) = 0;
    kernel = kernel(1:scale:end);
    kernel = kernel / sum(kernel);
    kernel(abs(kernel) < 0.01) = 0;
    deconv = deconvwnr(rfdata_obj.rfdata, kernel);
    %}

    %{
    kernel = kernel(1:scale:end);
    kernel = kernel / max(kernel);
    deconv = deconvwnr(rfdata_obj.rfdata, kernel);
    %}

    %%{
    kernel(1:22) = 0;%kernel(1:38) = 0;

    % kernel_fs = rfdata_obj.fs*scale;
    % k_filter_cutoff = [1.0]*1e6; % Hz
    % order = 5;
    % [b,a] = butter(order,k_filter_cutoff/(kernel_fs/2),'high');
    % kernel = filtfilt(b, a, kernel);

    kernel = imresize(kernel, 'Scale', [1/scale, 1], 'Method', 'lanczos3');
    kernel = kernel / sum(kernel);
    kernel(abs(kernel) < 0.05) = 0;
    kernel = kernel / sum(kernel);
    kernel = padarray(kernel, length(kernel), 0);
    pad_multiple = 1;
    deconv = deconvwnr(padarray(rfdata_obj.rfdata, [rfdata_obj.l_time*pad_multiple, 0, 0], 'circular', 'both'), kernel, estimated_nsr);
    deconv = deconv(rfdata_obj.l_time*pad_multiple:(end-(rfdata_obj.l_time*pad_multiple)), :, :);
    %}
    
    % deconv = deconvwnr(rfdata_obj.rfdata, kernel, estimated_nsr);
    toc
end