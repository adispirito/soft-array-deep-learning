
%% Load elem positions
params_ls = dir([params_dir,'*.mat']);
for iparam = 1:size(params_ls,1)
    params_name = params_ls(iparam).name;
    load([params_dir,strtrim(params_name)])
end
temperature_C = temperature_c;
sos = (1402.4+5.01*temperature_C-0.055*temperature_C^2+0.00022*temperature_C^3);
fs = Receive(1).decimSampleRate*1e6;
elemWidth = Trans.elementWidth; % mm
elemWidthwl = elemWidth/(sos/1e3/Trans.frequency);
wl = sos/Trans.frequency/1e3;
%% Define recon grid 

x_img = x_range(1):Res:x_range(2);
y_img = y_range(1):Res:y_range(2);
z_img = z_range(1):Res:z_range(2);
x_img0 = repmat(x_img',[1 length(y_img) length(z_img)]);
y_img0 = repmat(y_img,[length(x_img) 1 length(z_img)]);
z_img0 = repmat(z_img',[1 length(x_img) length(y_img)]);
z_img0 = permute(z_img0,[2 3 1]);

%% Process RF data
% TW.peak = 2.472;
RFshift = TW.peak/(Trans.frequency)*Receive(1).decimSampleRate; % TW.peak
RFData = imtranslate(RFData,[0 -RFshift],'FillValues',min(RFData(:)));

RFData = hilbert(RFData(:,:,iteststart:itestend));

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
%% TX delay mat
if ~exist('rTX','var')
    TXNum = size(TX_APOD,1);
    rTX = zeros(length(x_img),length(y_img),length(z_img),TXNum);
    tdend = 0;
    wb = waitbar(0,'Calculating RCV dist. map ...');
    txelemall = zeros(TXNum,1);
    for itx = 1:TXNum
        tdstart = tic;
        txelem = find(TX_APOD(itx,:)>0);
        txelemall(itx) = txelem;
        rTX(:,:,:,itx) = sqrt((x_img0 - x_trans(txelem)).^2 + ...
            (y_img0 - y_trans(txelem)).^2 + ...
            (z_img0 - z_trans(txelem)).^2);
        tdend = tdend + toc(tdstart);
        tdavg = tdend/itx;
        tdrem = (TXNum - itx)*tdavg;
        waitbar(itx/TXNum,wb,...
            [sprintf('%12.1f',tdrem) ' sec remaining calculating TX idx']);
    end
    close(wb)
    figure(4),scatter(x_trans,y_trans,'MarkerEdgeColor',[0 0.5 0.5],'LineWidth',1.5),hold on,
    scatter(x_trans(txelemall),y_trans(txelemall),'MarkerFaceColor',[0 0.7 0.7]),
    hold off, 
    %legend('All elems.','TX elems.'),xlabel('[mm]'),ylabel('[mm]'),
    pbaspect([1 1 1]),axis off
end
%% RCV delay mat, critical angle, angular sensitivity
if ~exist('rRCV','var')
    rRCV = zeros(length(x_img),length(y_img),length(z_img),Trans.numelements);
    % transSenAll = zeros(size(rRCV));
    elemAngleAll = zeros(1,Trans.numelements);
    tdend = 0;
    wb = waitbar(0,'Calculating RCV dist. map ...');
    for ielem = 1:Trans.numelements
        tdstart = tic;
        elemAngleAll(ielem) = abs(atan(sqrt(x_trans(ielem).^2 + y_trans(ielem).^2)/z_trans(ielem)));
        rRCV(:,:,:,ielem) = sqrt((x_img0 - x_trans(ielem)).^2 + ...
            (y_img0 - y_trans(ielem)).^2 + ...
            (z_img0 - z_trans(ielem)).^2);
        tdend = tdend + toc(tdstart);
        tdavg = tdend/ielem;
        tdrem = (Trans.numelements - ielem)*tdavg;
        waitbar(ielem/Trans.numelements,wb,...
            [sprintf('%12.1f',tdrem) ' sec remaining calculating RCV idx']);
    end
    close(wb)
end
%% Angular sensitivity
% cos(theta) = dot(u,v)/(norm(u)*norm(v));
% u: elem loc
% v: recon pixel -> elem
% transSenAll = single(ones(length(x_img),length(y_img),length(z_img),Trans.numelements));
if ~exist('transSenAll','var')
    angleAll = zeros(length(x_img),length(y_img),length(z_img),Trans.numelements);
    transSenAll = angleAll;
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
        transSenAll(:,:,:,ielem) = abs(cos(angleAll(:,:,:,ielem)).*...
            (sin(elemWidthwl*pi*sin(angleAll(:,:,:,ielem))))./...
            (elemWidthwl*pi*sin(angleAll(:,:,:,ielem))));
        taend = taend + toc(tastart);
        taavg = taend/ielem;
        tarem = (Trans.numelements - ielem)*taavg;
        waitbar(ielem/Trans.numelements,wb,...
            [sprintf('%12.1f',tarem) ' sec remaining calculating angular sensitivity']);
    end
    close(wb)
    clear x_img1 y_img1 z_img1 temp
end
%% Recon
framenum = size(RFData,3);

switch specialRecon
    case 'elemwise'
        us_rec_elem = zeros(length(x_img),length(y_img),length(z_img),Trans.numelements);
        framenum = 1;
        TXNum = 1;
        
    case 'txwise'
        us_rec_tx = zeros(length(x_img),length(y_img),length(z_img),framenum);
       
    case ''
        currTXNum = 1:TXNum;
end

us_rec = single(zeros(length(x_img),length(y_img),length(z_img),framenum));
RFData = single(RFData);
transSenAll = gpuArray(single(transSenAll));

trend = 0;
wb = waitbar(0,'Starting recon...');
for itx = currTXNum%1:TXNum
    trstart = tic;
    alineStart = Receive(itx).startSample;
    alineEnd = Receive(itx).endSample;
    iRF = gpuArray(single(RFData(alineStart:alineEnd,:,:)));
    idxAll = gpuArray(single(round((repmat(rTX(:,:,:,itx),[1 1 1 Trans.numelements]) + rRCV - 2*P.acqStartDepth*wl)/1e3/sos*fs)));
    
    for iframe = 1:framenum
        us_bmode = gpuArray(single(zeros(size(us_rec(:,:,:,1)))));
        for ielem = 1:Trans.numelements
            [us_bmode,us_bmode1] = GPUReconLoop(iRF,idxAll,iframe,ielem,us_bmode,transSenAll);
            if strcmp(specialRecon,'elemwise')
                us_rec_elem(:,:,:,ielem) = gather(us_bmode1);
            end
        end
        %     pa_rec(:,:,:,iframe) = gather(pa_bmode);
        us_rec(:,:,:,iframe) = us_rec(:,:,:,iframe) + single(gather(us_bmode));
        
        if strcmp(specialRecon,'txwise')
            us_rec_tx(:,:,:,iframe) = single(gather(us_bmode));
        end
    end
    
    trend = trend + toc(trstart);
    travg = trend/itx;
    trrem = (TXNum - itx)*travg;
    waitbar(itx/TXNum,wb,...
        [sprintf('%12.1f',trrem) ' sec remaining for recon']);
end
close(wb)
us_rec = single(us_rec);
%%
xymap = squeeze(max(abs(mean(us_rec(:,:,:,:),4)),[],3));
xzmap = squeeze(max(abs(mean(us_rec(:,:,:,:),4)),[],1))';
yzmap = squeeze(max(abs(mean(us_rec(:,:,:,:),4)),[],2))';
figure(1),
subplot(1,3,1),imagesc(x_img,y_img,20*log10(xymap/max(xymap(:))))
colormap gray, caxis(DISPCAXIS)
pbaspect([length(x_img)/length(y_img) 1 1]),xlabel('x [mm]'),ylabel('y [mm]')

subplot(1,3,2),imagesc(x_img,z_img,20*log10(xzmap/max(xzmap(:))))
colormap gray, caxis(DISPCAXIS)
pbaspect([length(y_img)/length(z_img) 1 1]),xlabel('y [mm]'),ylabel('z [mm]')

subplot(1,3,3),imagesc(y_img,z_img,20*log10(yzmap/max(yzmap(:))))
colormap gray, caxis(DISPCAXIS)
pbaspect([length(x_img)/length(z_img) 1 1]),xlabel('x [mm]'),ylabel('z [mm]')

%%
diginum = 4;
fprintf('Saving...\n')
if ~RTSAVING
    save_dir1 = save_dir;
    idatfilesname = strtrim(file_names(ifile,:));
    ibatchstart = TESTstart;
    ibatchend = TESTend;
else
    save_dir1 = [save_dir,folder_names(ifolder).name,'\'];
end
if SAVE_DATA
    checkMakeDir(save_dir1)
    if exist('us_rec_elem','var') %&& max(us_rec_elem(:)) ~= 0
        save_name = [strtrim(idatfilesname(1:end-4)),'_',num2str(sprintf(['%0',num2str(diginum),'d'],ibatchstart)),'_',...
            num2str(sprintf(['%0',num2str(diginum),'d'],ibatchend)),'_tC_',num2str(temperature_C),'_recon_elemwise'];
        resultpara_name = [save_name(1:end),'_param'];
        
        tic
        save([save_dir1,resultpara_name,'.mat'],'x_img','y_img',... 
            'z_img','temperature_C','sos')
        f1 = fopen([save_dir1,save_name,'.dat'],'w');
        us_rec_elem0 = single([reshape(real(us_rec_elem),[],1);reshape(imag(us_rec_elem),[],1)]);
        fwrite(f1,us_rec_elem0,'single');
        fclose(f1);
        toc
    elseif exist('us_rec_tx','var') %%max(us_rec_tx(:)) ~= 0
        save_name = [strtrim(idatfilesname(1:end-4)),'_',num2str(sprintf(['%0',num2str(diginum),'d'],ibatchstart)),'_',...
            num2str(sprintf(['%0',num2str(diginum),'d'],ibatchend)),'_tC_',num2str(temperature_C),'_recon_txwise',num2str(currTXNum)];
        resultpara_name = [save_name(1:end),'_param'];
        
        tic
        save([save_dir1,resultpara_name,'.mat'],'x_img','y_img',...
            'z_img','temperature_C','sos')
        f1 = fopen([save_dir1,save_name,'.dat'],'w');
        us_rec_tx0 = single([reshape(real(us_rec_tx),[],1);reshape(imag(us_rec_tx),[],1)]);
        fwrite(f1,us_rec_tx0,'single');
        fclose(f1);
        toc
    else
        save_name = [strtrim(idatfilesname(1:end-4)),'_',num2str(sprintf(['%0',num2str(diginum),'d'],ibatchstart)),'_',...
            num2str(sprintf(['%0',num2str(diginum),'d'],ibatchend)),'_tC_',num2str(temperature_C),'_recon'];
        resultpara_name = [save_name(1:end),'_param'];
        tic
        save([save_dir1,resultpara_name,'.mat'],'x_img','y_img',...
            'z_img','temperature_C','sos','x_range','y_range','z_range')
        f1 = fopen([save_dir1,save_name,'.dat'],'w');
        if ABSDATA
            us_rec0 = single(reshape(abs(us_rec),[],1));
        else
            us_rec0 = single([reshape(real(us_rec),[],1);reshape(imag(us_rec),[],1)]);
        end
        fwrite(f1,us_rec0,'single');
        fclose(f1);
        toc
    end
end
%%
function [us_bmode,us_bmode1] = GPUReconLoop(rfdataAll,idxAll,iframe,iline,us_bmode,transSenAll)
if iline == 1, us_bmode = 0*us_bmode; end
rfdata = rfdataAll(:,iline,iframe);
us_bmode1 = interp1(rfdata,idxAll(:,:,:,iline));
us_bmode = us_bmode + us_bmode1.*transSenAll(:,:,:,iline);
end