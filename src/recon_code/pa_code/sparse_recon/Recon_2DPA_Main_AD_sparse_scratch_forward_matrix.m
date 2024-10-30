RFDataCls = @sparse_recon.classes.RFDataCls;
IndexMatrix2D = @sparse_recon.classes.IndexMatrix2D;
RFInterpolantCls = @sparse_recon.classes.RFInterpolantCls;
RFIDWCls = @sparse_recon.classes.RFIDWCls;
%% Determine p0
p0 = zeros(l_xarr, l_yarr, l_zarr);
p0(:, 33, 33) = 1;
p0(33, :, 33) = 1;
p0 = smooth(p0);

% p0 = ones(l_xarr, l_yarr, l_zarr);

rfdata = forward_mat*p0(:);
rfdata = reshape(rfdata, [l_time, num_elem]);
% rfdata = padarray(rfdata, 1900-size(rfdata, 1), 0, "post");
rfdata = repmat(rfdata, 1, 1, 10);
%% Set Sensor Array

HILBERTRF = 0;
INTERP = 'time';%'';
TIME_INTERP_RATIO = 10;
FILTER = '';
GPU_ID = 1;
SensorArray2D = sensorarr;
fs = target_fs;
Trans.elementWidth = 0; % placeholder - is not used
Trans.frequency = fs/4;
RFDataObj = RFDataCls(rfdata, fs, sos, Trans);

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

%% Interpolation
%%{
fprintf("Array Size (Before Interpolation): (%i, %i, %i)\n", size(RFDataObj.rfdata))
%INTERP = "time";%"time";%"fast_channels+time";
switch INTERP
    case "fast_channels+time"
        % Time Interpolation
        RFDataObj.rfdata = imresize3(RFDataObj.rfdata, "Scale", ...
                                     [TIME_INTERP_RATIO, 1, 1], ...
                                     "Method", 'cubic'); % linear / cubic / lanczos3
        RFDataObj.fs = RFDataObj.fs*TIME_INTERP_RATIO;
        % RFDataObj.l_time = RFDataObj.l_time*TIME_INTERP_RATIO;
        
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
        z_trans = interp(z_trans_orig, INTERP_RATIO);

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
        % RFDataObj.num_elements = RFDataObj.num_elements * INTERP_RATIO;
    case "time"
        RFDataObj.rfdata = imresize3(RFDataObj.rfdata, "Scale", [TIME_INTERP_RATIO, 1, 1], "Method", 'cubic'); % linear / cubic / lanczos3
        RFDataObj.fs = RFDataObj.fs*TIME_INTERP_RATIO;
        % RFDataObj.l_time = RFDataObj.l_time*TIME_INTERP_RATIO;
end
%}
%% Build Sparse Matrix
USE_GPU = 0;
fprintf("Array Size (After Interpolation): (%i, %i, %i)\n", size(RFDataObj.rfdata))
fprintf('Use GPU = %d\n', USE_GPU)
fprintf('Creating Sparse Matrix... \n')
if USE_GPU
    tic
    IndexMatrix2D = IndexMatrix2D(SensorArray2D, recongrid, RFDataObj, "fast_matrix_gpu", GPU_ID);
    toc
    disp(size(IndexMatrix2D.M))
    disp(size(RFDataObj.rfdata))
    out_size = [RFDataObj.l_time*RFDataObj.num_elements, RFDataObj.num_frames];
    RFDataObj.rfdata = reshape(RFDataObj.rfdata, out_size);
    RFDataObj.rfdata = double(RFDataObj.rfdata);
    rfdata = gpuArray(RFDataObj.rfdata);
    % RFDataObj = RFDataObj.to_gpu(GPU_ID);
else
    tic
    IndexMatrix2D = IndexMatrix2D(SensorArray2D, recongrid, RFDataObj, "fast_matrix", GPU_ID);
    toc
    disp(size(IndexMatrix2D.M))
    disp(size(RFDataObj.rfdata))
    out_size = [RFDataObj.l_time*RFDataObj.num_elements, RFDataObj.num_frames];
    rfdata = reshape(RFDataObj.rfdata, out_size);
end
fprintf('Performing Sparse Matrix Matrix Multi... ')
tic
pa_rec = IndexMatrix2D.M*rfdata;
pa_rec = reshape(pa_rec, [recongrid.grid_size(:)', RFDataObj.num_frames]);
toc
disp("Done!!!")


%% Convert to single
pa_rec = single(pa_rec);

%% Remove Breathing Frames
if RM_BREATH
    vol_size = size(pa_rec);
    corrs = [];
    for i = 1:vol_size(end)
        vol1 = pa_rec(:, :, :, 1);
        vol2 = pa_rec(:, :, :, i);
        corre = corrcoef(vol1(:), vol2(:));
        corrs(i) = corre(1, 2);
    end
    frames = corrs >= CORR_THRES;
    pa_rec = pa_rec(:, :, :, frames);
end
%{
%%
if isreal(pa_rec)
    xymap = squeeze(max(mean(pa_rec,4),[],3));
    xzmap = squeeze(max(mean(pa_rec,4),[],1))';
    yzmap = squeeze(max(mean(pa_rec,4),[],2))';
else
    xymap = squeeze(max(abs(mean(pa_rec,4)),[],3));
    xzmap = squeeze(max(abs(mean(pa_rec,4)),[],1))';
    yzmap = squeeze(max(abs(mean(pa_rec,4)),[],2))';
end
%%
%{
DISPCAXIS = [min(abs(pa_rec(:))) max(abs(pa_rec(:)))];
figure(1)
subplot(1,3,1),imagesc(x_img,y_img,xymap)
colormap hot, caxis(DISPCAXIS)
pbaspect([length(x_img)/length(y_img) 1 1]),xlabel('x [mm]'),ylabel('y [mm]')
colorbar
subplot(1,3,2),imagesc(x_img,z_img,xzmap)
colormap hot, caxis(DISPCAXIS)
pbaspect([length(x_img)/length(z_img) 1 1]),xlabel('x [mm]'),ylabel('z [mm]')
colorbar
subplot(1,3,3),imagesc(y_img,z_img,yzmap)
colormap hot, caxis(DISPCAXIS)
pbaspect([length(y_img)/length(z_img) 1 1]),xlabel('y [mm]'),ylabel('z [mm]')
colorbar
%}
%%
% xymap1 = xymap/max(xymap(:));xymap1(xymap1<0) = 0; 
% figure(1),imagesc(x_img,y_img,20*log10(xymap1))
% colormap hot, caxis(DISPCAXIS)
% pbaspect([length(x_img)/length(y_img) 1 1]),xlabel('x [mm]'),ylabel('y [mm]')
% 
% xzmap1 = xzmap/max(xzmap(:));xzmap1(xzmap1<0) = 0; 
% figure(2),imagesc(x_img,z_img,20*log10(xzmap1))
% colormap hot, caxis(DISPCAXIS)
% pbaspect([length(x_img)/length(z_img) 1 1]),xlabel('x [mm]'),ylabel('z [mm]')
% 
% yzmap1 = yzmap/max(yzmap(:));yzmap1(yzmap1<0) = 0; 
% figure(3),imagesc(y_img,z_img,20*log10(yzmap1))
% colormap hot, caxis(DISPCAXIS)
% pbaspect([length(y_img)/length(z_img) 1 1]),xlabel('y [mm]'),ylabel('z [mm]')
%
%%
diginum = 4;
if SAVE_DATA
    tic
%     save_paramname = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
%         num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_params.mat'];
    save_name = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
        num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_recon'];
    if DOWNSAMPLE
        save_name = [save_name, '_down', down_arr_filename(end-6:end-4)];
    end
    if save_dir(end) ~= '\'
        save_dir = [save_dir, '\'];
    end
    checkMakeDir(save_dir)
%     checkMakeDir([save_dir,folder_names(ifolder).name,'\'])
%     save([save_dir,folder_names(ifolder).name,'\',save_paramname],'x_img','y_img',...
%         'z_img','temperature_C','x_range','y_range','z_range',...
%         'laser_type','flash2Qdelay','sos','HILBERTRF','MULTI_SHIFT');
    
%     f1 = fopen([save_dir,folder_names(ifolder).name,'\',save_name,'.dat'],'w');
%     if ABSDATA && HILBERTRF
%         pa_rec0 = single(reshape(abs(pa_rec),[],1));
%     elseif HILBERTRF
%         pa_rec0 = single([reshape(real(pa_rec),[],1);reshape(imag(pa_rec),[],1)]);
%     else
%         pa_rec0 = single(reshape(pa_rec,[],1));
%     end
    if ABSDATA && HILBERTRF
        pa_rec0 = single(abs(pa_rec));
    elseif HILBERTRF
        pa_rec0 = single(cat(5, real(pa_rec),imag(pa_rec)));
    else
        pa_rec0 = single(pa_rec);
    end
    if SAVE_AVG
        pa_rec0 = squeeze(mean(pa_rec0, 4));
    end
%     fwrite(f1,pa_rec0,'single');
%     fclose(f1);
    save([save_dir, save_name], 'pa_rec0','x_img','y_img',...
        'z_img','temperature_C','x_range','y_range','z_range',...
        'laser_type','flash2Qdelay','sos','HILBERTRF','MULTI_SHIFT', ...
        '-v7.3', '-nocompression');
    toc
end
%% Subfunctions
function checkMakeDir(path)
    if exist(path,'dir') 
        fprintf('Saving path already exists.\n')
    else
        mkdir(path);
    end
end
%}