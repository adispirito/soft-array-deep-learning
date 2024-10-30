function Recon_2DPA_Drive_AD_func(save_sub_dir, SENSOR_POS_UNCERTAIN, STD_ERR, GPU_ID)
    arguments
        save_sub_dir
        SENSOR_POS_UNCERTAIN {mustBeInteger} = 0;
        STD_ERR {mustBeNumeric} = 0;
        GPU_ID {mustBeInteger} = -1;
    end
    %% Recon_PA_2D
    % clear,
    % clc, close all
    %g = gpuDevice(2);
    % reset(g);
    
    %%
    %root_dir = 'C:\Users\PI-Lab\Desktop\DOCUMENTS\Anthony\PACT\SparseArray\soft-array-deep-learning\src\';
    root_dir = 'D:\Anthony\PACT\SparseArray\soft-array-deep-learning\src\';
    file_dir = [root_dir, 'recon_code\data\raw_data\'];
    %save_sub_dir = 'recon_code\data\raw_mat\sensor_pos_err_quant\std_err_0-00000';
    save_dir = [root_dir, save_sub_dir];
    code_dir = './';
    params_dir = './+sparse_recon/params/';%'../recon_params/';%'./+sparse_recon/params/';
    down_arr_filename = 'undersample_map128.mat';
    
    %% file selection, recon param, and flag
    
    [HILBERTRF, ABSDATA] = deal(0, 0);
    RTSAVING = 1;
    DECONVOLVE = 0; % use deconvwnr
    ENERGY_NORM = 0;
    SAVE_DATA = 1;
    RM_BREATH = 1;
    CORR_THRES = 0.9;
    SAVE_AVG = 1;
    DOWNSAMPLE = 0;
    % [INTERP, INTERP_RATIO] = deal(1, 3);
    % [INTERP, INTERP_RATIO] = deal("fast_channels+time", 2); % "simple_channels+time" / "channels+time" / "time"
    %[INTERP, INTERP_RATIO, TIME_INTERP_RATIO] = deal("fast_channels+time", 3, 4);
    [INTERP, INTERP_RATIO, TIME_INTERP_RATIO] = deal("time", 1, 3);
    % SENSOR_POS_UNCERTAIN = 0;
    %STD_ERR = 0.00001;
    
    Res = 0.06;%0.06;%0.06;
    cAngle = 60*pi/180;
    x_range = [-4.1,4];
    y_range = [-4,4.2];
    z_range = [-4,4];
    
    FILTER = 'BP';%'';
    FILTERCUTOFF = [0.5 7]*1e6;%[7]*1e6;%[1]*1e6; % Hz
    FILTERRFCOMP = 0;
    ORDER = 6;
    
    FRAMESTART = 1;
    FOLDERSELECTION = {'hair'};%{'PSF'};%{'hair'};
    DISPCAXIS = [0 1]*1e4;
    % DISPCAXIS = [-20 0];
    
    MULTI_WL = 3; % number of wl, if > 0, enter the number of wl used
    RECON_WL = 1:2;%2;%1:2;%2;%1:2;
    MULTI_SOS = 0;
    SOS = 21:25;
    % MULTI_SHIFT = [0 35+1.78 230+1.14]; %532 1064,190+35,195+35+205, micro sec
    MULTI_SHIFT = [0 0 0];
    % RF_SHIFT = [14 36 42]; % OPO: 6,
    RF_SHIFT = [8 41 45]; % MEASURED VALUE, DONT CHANGE, FOR OPO, 1064, 532
    BATCHSIZE = 10;%28;%109;%10;%28;
    %% For realtime saving
    if RTSAVING
        folder_names = dir(file_dir);
        folder_flags = [folder_names.isdir];
        folder_names = folder_names(folder_flags);
        folder_names = folder_names(3:end,:); file_names = folder_names;
        folder_num = size(folder_names,1);
        fprintf([num2str(folder_num),' folders detected in the directory.\n']);
        for ifolder = 1:folder_num
            currentfolder = [file_dir,folder_names(ifolder).name,'\'];
            % check if folder satisfy the criteria
            criteriacheck = 0;
            for icriteria = 1:length(FOLDERSELECTION)
                if contains(folder_names(ifolder).name,char(FOLDERSELECTION(icriteria)))
                    criteriacheck = criteriacheck + 1;
                end
            end
            if criteriacheck < length(FOLDERSELECTION)
                continue;
            end
            % load Setup parameters (Saved Acquisition Config)
            parafile = dir(fullfile(currentfolder,'*.mat'));
            parafile = parafile.name;
            load([currentfolder,parafile]);
            disp(['Current folder: ',currentfolder]);
            % define batch size and start/end index for each batch
    
            DATNUM = size(dir(fullfile(currentfolder,'*.dat')),1);
            if rem(DATNUM,BATCHSIZE) == 0
                TESTstart = FRAMESTART:BATCHSIZE:DATNUM;
                TESTend = BATCHSIZE:BATCHSIZE:DATNUM;
            else
                TESTstart = 1:BATCHSIZE:DATNUM;
                TESTend = BATCHSIZE:BATCHSIZE:DATNUM;
                TESTend = [TESTend DATNUM];
            end
            
            % Energy Normalization
            if ENERGY_NORM
                num_lasers = MULTI_WL;
                channel_rec = 1;
                energyfile = dir(fullfile(currentfolder,'*.txt'));
                [ch_data,~] = readLogStarlab2D([currentfolder,energyfile.name]);
                singleCh_data = reshape(ch_data(:,channel_rec*2),num_lasers,...
                    length(ch_data)/num_lasers,[])';
                first_val = zeros(1,num_lasers);
                for ilaser = 1:num_lasers
                    isingleCh = singleCh_data(:,ilaser);
                    if ~isempty(find(isingleCh > 0,1)) && mode(isingleCh) > 0
                        first_val(ilaser) = isingleCh(find(isingleCh > 0,1));
                    else
                        fprintf(['No energy recording for wavelength #',num2str(ilaser),'\n'])
                        singleCh_data(:,ilaser) = 1;
                        first_val(ilaser) = 1;
                    end
                end
                singleCh_norm = singleCh_data./first_val;
                singleCh_norm(singleCh_norm == 0) = 1;
            end
            
            % load batch data
            for iReconBatch = 1:length(TESTstart)
                iteststart = TESTstart(iReconBatch);
                itestend = TESTend(iReconBatch);
                fprintf(['Current recon frames: ',num2str(iteststart),'-',...
                    num2str(itestend), ' out of ', num2str(DATNUM),'\n']);
                datfiles = dir(fullfile(currentfolder,'*.dat'));
                datfileind = iteststart:itestend;
                
                RFData = zeros(RFDataSize(1),RFDataSize(2),itestend-iteststart+1);
                for idatfile = 1:(itestend-iteststart+1)
                    idatfilesname = datfiles(datfileind(idatfile)).name;
                    f = fopen([currentfolder,idatfilesname],'r');
                    iRFData = fread(f,'int16');
                    iRFData = reshape(iRFData,[RFDataSize(1) RFDataSize(2)]);
                    fclose(f);
                    RFData(:,:,idatfile) = iRFData;
                end % finish loading file
                
                RFData0 = RFData;
                file = [folder_names(ifolder).name,'....'];
                % Downsample Channels
                if DOWNSAMPLE
                    sample_mask = load([code_dir, down_arr_filename]);
                    sample_mask = sample_mask.sample;
                    RFData0(:, sample_mask, :) = 0;
                end
                
                for iwl = RECON_WL
                    if MULTI_SOS
                        for isos = 1:length(SOS)
                            RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
                            clear angleAll idxAll
                            temperature_C = SOS(isos);
                            
                            fprintf(['Current wl: ',num2str(iwl),'/',...
                                num2str(MULTI_WL),'\n','Current tC: ',num2str(temperature_C),'\n']);
                            run sparse_recon/Recon_2DPA_Main_AD_sparse_scratch
                            % run Recon_2DPA_Main_AD
                        end
                    else
                        RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
    %                     clear angleAll idxAll
                        
                        fprintf(['Current wl: ',num2str(iwl),'/',...
                            num2str(MULTI_WL),'\n']);
                        run sparse_recon/Recon_2DPA_Main_AD_sparse_scratch
                        % run Recon_2DPA_Main_AD
                        %run('./orig_code/Recon_2DPA_Main')
                    end
                end
                         
            end
        end
    %{    
    else
        %For batch saving
        file_names = batchReader(file_dir,'*.mat',{'PA'});
        TESTstart = 1:200:400;
        TESTend = 200:200:400;
        
        for ifile = 1:size(file_names,1)
            for iReconBatch = 1:length(TESTstart)
                close all
                if MULTI_WL == 2
                    for iwl = 1:2
                        file = strtrim(file_names(ifile,:))
                        load([file_dir,file]);
                        
                        iteststart = TESTstart(iReconBatch);
                        itestend = TESTend(iReconBatch);
                        RFData = RcvData{1}; clear RcvData
                        RFData = RFData(Receive(2).startSample:Receive(2).endSample,:,iteststart:itestend);
                        run Recon_2DPA_Main_AD
                    end
                else
                    iwl = 1;
                    file = strtrim(file_names(ifile,:))
                    load([file_dir,file]);
                    
                    iteststart = TESTstart(iReconBatch);
                    itestend = TESTend(iReconBatch);
                    RFData = RcvData{1}; clear RcvData
                    RFData = RFData(Receive(2).startSample:Receive(2).endSample,:,iteststart:itestend);
                    run Recon_2DPA_Main_AD
                end
            end
        end
        %}
    end
end