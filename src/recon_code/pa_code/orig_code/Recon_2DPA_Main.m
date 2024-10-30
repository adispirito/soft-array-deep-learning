
%% Load elem positions
params_ls = dir([params_dir,'*.mat']);
for iparam = 1:size(params_ls,1)
    params_name = params_ls(iparam).name;
    load([params_dir,strtrim(params_name)])
end

sos = (1402.4+5.01*temperature_C-0.055*temperature_C^2+0.00022*temperature_C^3);
fs = Receive(1).decimSampleRate*1e6;
elemWidth = Trans.elementWidth; % mm
elemWidthwl = elemWidth/(sos/1e3/Trans.frequency);

%% Define recon grid 
x_img = x_range(1):Res:x_range(2);
y_img = y_range(1):Res:y_range(2);
z_img = z_range(1):Res:z_range(2);
x_img0 = repmat(x_img',[1 length(y_img) length(z_img)]);
y_img0 = repmat(y_img,[length(x_img) 1 length(z_img)]);
z_img0 = repmat(z_img',[1 length(x_img) length(y_img)]);
z_img0 = permute(z_img0,[2 3 1]);

%% Process RF data
if ENERGY_NORM
    fprintf('RF Data Energy Normalization: ')
    tic
    ibatchEnergy = singleCh_norm(iteststart:itestend,iwl);
    RFData = RFData./reshape(ibatchEnergy,1,1,[]); 
    toc
end

if DECONVOLVE
    fprintf('RF Data Deconvolution: ')
    tic
    RFData = deconvwnr(RFData,IR1Way(1:4:end));
    toc
    deconshift = 5;
else
    deconshift = 0;
end

RFData = imtranslate(RFData,[0 -RF_SHIFT(iwl)-deconshift],'FillValues',min(RFData(:)));

switch FILTER
    case 'HP'
        fprintf('HPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(fs/2),'high');
    case 'LP'
        fprintf('LPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(fs/2),'low');
    case 'BP'
        fprintf('BPF applied. \n')
        [b,a]    = butter(ORDER,FILTERCUTOFF/(fs/2),'bandpass');
end

if ~isempty(FILTER)
   fprintf('Filtering: ')
   tic
   RFData = imtranslate(filter(b,a,RFData,[],1),[0 -FILTERRFCOMP],'FillValues',min(RFData(:)));
   toc
end

if HILBERTRF
    fprintf('RF Data Hilbert Transform: ')
    tic
    RFData = hilbert(RFData);
    toc
end
%% Angular sensitivity
% cos(theta) = dot(u,v)/(norm(u)*norm(v));
% u: elem loc
% v: recon pixel -> elem
if ~exist('angleAll','var')
    angleAll = zeros(length(x_img),length(y_img),length(z_img),Trans.numelements);
    [xsize,ysize,zsize] = size(x_img0);
    x_img1 = reshape(x_img0,[xsize*ysize*zsize,1]);
    y_img1 = reshape(y_img0,[xsize*ysize*zsize,1]);
    z_img1 = reshape(z_img0,[xsize*ysize*zsize,1]);
    
    taend = 0;
    wb = waitbar(0,'Calculating angular sensitivity map ...');
    for ielem = 1:Trans.numelements
        tastart = tic;
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
        taend = taend + toc(tastart);
        taavg = taend/ielem;
        tarem = (Trans.numelements - ielem)*taavg;
        waitbar(ielem/Trans.numelements,wb,...
            [sprintf('%12.1f',tarem) ' sec remaining calculating angular sensitivity']);
    end
    close(wb)
    clear x_img1 y_img1 z_img1 temp
end
%% Delay mat, critical angle, angular sensitivity
if ~exist('idxAll','var')
    idxAll = zeros(length(x_img),length(y_img),length(z_img),Trans.numelements);
    % angleAll = zeros(size(idxAll));
    transSenAll = zeros(size(idxAll));
    elemAngleAll = zeros(1,Trans.numelements);
    tdend = 0;
    wb = waitbar(0,'Calculating dist. map ...');
    for ielem = 1:Trans.numelements
        tdstart = tic;
        elemAngleAll(ielem) = abs(atan(sqrt(x_trans(ielem).^2 + y_trans(ielem).^2)/z_trans(ielem)));
        idxAll(:,:,:,ielem) = (sqrt((x_img0 - x_trans(ielem)).^2 + ...
            (y_img0 - y_trans(ielem)).^2 + ...
            (z_img0 - z_trans(ielem)).^2)/1e3/sos)*fs;
        %     angleAll(:,:,:,ielem) = abs(atan(abs(sqrt((x_img0 - x_trans(ielem)).^2 +...
        %         (y_img0 - y_trans(ielem)).^2)./(z_img0 - z_trans(ielem))))-elemAngleAll(ielem));
        transSenAll(:,:,:,ielem) = abs(cos(angleAll(:,:,:,ielem)).*...
            (sin(elemWidthwl*pi*sin(angleAll(:,:,:,ielem))))./...
            (elemWidthwl*pi*sin(angleAll(:,:,:,ielem))));
        tdend = tdend + toc(tdstart);
        tdavg = tdend/ielem;
        tdrem = (Trans.numelements - ielem)*tdavg;
        waitbar(ielem/Trans.numelements,wb,...
            [sprintf('%12.1f',tdrem) ' sec remaining calculating pixel position']);
    end
    close(wb)
    idxAll = round(idxAll);
end
% inrange = (idxAll > 1) & (angleAll <= cAngle); 
% clear angleAll
%% Recon
framenum = size(RFData,3);
pa_rec = zeros(length(x_img),length(y_img),length(z_img),framenum);
RFData = gpuArray(single(RFData));
idxAll = gpuArray(single(idxAll));
% inrange = gpuArray(inrange);
transSenAll = gpuArray(single(transSenAll));
trend = 0;
wb = waitbar(0,'Starting recon...');
elems = 1:Trans.numelements;
for iframe = 1:framenum
    trstart = tic;
    pa_bmode = zeros(size(idxAll(:,:,:,1)));
    pa_bmode = gpuArray(single(pa_bmode));
    for ielem = 1:Trans.numelements
        %         pa_bmode = GPUReconLoop(RFData,idxAll,iframe,ielem,pa_bmode,inrange,transSenAll);
        pa_bmode = GPUReconLoop(RFData,idxAll,iframe,ielem,pa_bmode,transSenAll);
    end
    pa_rec(:,:,:,iframe) = gather(pa_bmode);
%     pa_rec(:,:,:,iframe) = pa_bmode;
    
    trend = trend + toc(trstart);
    travg = trend/iframe;
    trrem = (size(RFData,3) - iframe)*travg;
    waitbar(iframe/framenum,wb,...
        [sprintf('%12.1f',trrem) ' sec remaining for recon']);
end
close(wb)
%% if HILBERTRF
pa_rec = single(pa_rec);
% end
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
    save_paramname = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
        num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_params.mat'];
    save_name = [file(1:end-4),'_wl',num2str(iwl),'_',num2str(sprintf(['%0',num2str(diginum),'d'],iteststart)),'_',...
        num2str(sprintf(['%0',num2str(diginum),'d'],itestend)),'_tC_',num2str(temperature_C),'_recon'];
    
    checkMakeDir([save_dir,folder_names(ifolder).name,'\'])
    save([save_dir,folder_names(ifolder).name,'\',save_paramname],'x_img','y_img',...
        'z_img','temperature_C','x_range','y_range','z_range',...
        'laser_type','flash2Qdelay','sos','HILBERTRF','MULTI_SHIFT');
    
    f1 = fopen([save_dir,folder_names(ifolder).name,'\',save_name,'.dat'],'w');
    if ABSDATA && HILBERTRF
        pa_rec0 = single(reshape(abs(pa_rec),[],1));
    elseif HILBERTRF
        pa_rec0 = single([reshape(real(pa_rec),[],1);reshape(imag(pa_rec),[],1)]);
    else
        pa_rec0 = single(reshape(pa_rec,[],1));
    end
    fwrite(f1,pa_rec0,'single');
    fclose(f1);
    toc
end
%%
function pa_bmode = GPUReconLoop(rfdataAll,idxAll,iframe,iline,pa_bmode,transSenAll)
if iline == 1, pa_bmode = 0*pa_bmode; end
rfdata = rfdataAll(:,iline,iframe);
pa_bmode1 = interp1(rfdata,idxAll(:,:,:,iline));
% pa_bmode = pa_bmode + pa_bmode1.*inrange(:,:,:,iline).*transSenAll(:,:,:,iline);
pa_bmode = pa_bmode + pa_bmode1.*transSenAll(:,:,:,iline);
end