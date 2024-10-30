%% Recon_PA_2D
clear,
% clc, close all
g = gpuDevice(1);
reset(g);

%%
ctime = clock;
timestamp = datestr(datenum(ctime),'yyyymmddTHHMMSS');
exp_date = '20221025 hair';
SAVESUFFIX = '_PA';
DATALOC = 5;
switch DATALOC
    case 1
        file_dir = ['J:\01_Yuqi\Data\',exp_date,'\'];
    case 2
        file_dir = ['G:\YuqiStorage\1Data\',exp_date,'\'];
    case 3
        file_dir = ['D:\Lab Members\Yuqi\Local Data\',exp_date,'\'];
    case 4
        file_dir = ['A:\Yuqi Data Storage\1Data\',exp_date,'\'];
    case 5
        file_dir = ['C:\Users\PI-Lab\Desktop\DOCUMENTS\Anthony\PACT\SparseArray\dl_code\src\recon_code\data\raw_data\'];
end
save_dir = 'C:\Users\PI-Lab\Desktop\DOCUMENTS\Anthony\PACT\SparseArray\dl_code\src\recon_code\data\raw_mat\';
code_dir = './';
params_dir = './Params/';

%% file selection, recon param, and flag

HILBERTRF = 0;
ABSDATA = 0;
RTSAVING = 1;
DECONVOLVE = 0; % use deconvwnr
ENERGY_NORM = 0;
SAVE_DATA = 0;

Res = 0.06;
cAngle = 60*pi/180;
x_range = [-4.1,4];
y_range = [-4,4.2];
z_range = [-4,4];

FILTER = '';
FILTERCUTOFF = [1]*1e6;
FILTERRFCOMP = 0;
ORDER = 6;

FRAMESTART = 1;
FOLDERSELECTION = {'Blvra','liver', '2#'};
DISPCAXIS = [0 1]*1e4;
% DISPCAXIS = [-20 0];

MULTI_WL = 3; % number of wl, if > 0, enter the number of wl used
RECON_WL = 1;
MULTI_SOS = 0;
SOS = 21:25;
% MULTI_SHIFT = [0 35+1.78 230+1.14]; %532 1064,190+35,195+35+205, micro sec
MULTI_SHIFT = [0 0 0];
% RF_SHIFT = [14 36 42]; % OPO: 6,
RF_SHIFT = [8 41 45]; % MEASURED VALUE, DONT CHANGE, FOR OPO, 1064, 532
BATCHSIZE = 60;
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
        % load parameters
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
        
        % Energy Normalizationn
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
            
            for iwl = RECON_WL
                if MULTI_SOS
                    for isos = 1:length(SOS)
                        RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
                        clear angleAll idxAll
                        temperature_C = SOS(isos);
                        
                        fprintf(['Current wl: ',num2str(iwl),'/',...
                            num2str(MULTI_WL),'\n','Current tC: ',num2str(temperature_C),'\n']);
                        run Recon_2DPA_Main
                    end
                else
                    RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
%                     clear angleAll idxAll
                    
                    fprintf(['Current wl: ',num2str(iwl),'/',...
                        num2str(MULTI_WL),'\n']);
                    run Recon_2DPA_Main
                end
            end
                     
        end
    end
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
                    run Recon_2DPA_Main
                end
            else
                iwl = 1;
                file = strtrim(file_names(ifile,:))
                load([file_dir,file]);
                
                iteststart = TESTstart(iReconBatch);
                itestend = TESTend(iReconBatch);
                RFData = RcvData{1}; clear RcvData
                RFData = RFData(Receive(2).startSample:Receive(2).endSample,:,iteststart:itestend);
                run Recon_2DPA_Main
            end
        end
    end
end