%% Recon_PA_2D
clear, clc, 
%close all
g = gpuDevice(1);
reset(g);

%%
ctime = clock;
timestamp = datestr(datenum(ctime),'yyyymmddTHHMMSS');
exp_date = '20221025 hair';
SUFFIX = 'US';
DATALOC = 2;
switch DATALOC
    case 1
        file_dir = ['J:\01_Yuqi\Data\',exp_date,'\'];
    case 2
        file_dir = ['G:\YuqiStorage\1Data\',exp_date,'\'];
    case 3
        file_dir = ['D:\Lab Members\Yuqi\Local Data\',exp_date,'\'];
    case 4
        file_dir = ['A:\Yuqi Data Storage\1Data\',exp_date,'\'];
end
save_dir = ['G:\YuqiStorage\2Results\',exp_date,'\',timestamp,'_',SUFFIX,'\'];
code_dir = 'D:\Lab Members\Yuqi\Dropbox\0Code\';
params_dir = 'D:\Lab Members\Yuqi\Dropbox\0Code\2D\SetupScripts\Params\';
%% file selection, recon param, and flag
                                                                                                                                                               
subgroup = 10;
groupnum = 1;
CUTDATA = 0;
HALFWAY = 0;
ABSDATA = 1;
HILBERTRF = 1;

BATCHTRANSFER = 1;
RTSAVING = 1;
FOLDERSELECTION = {'US'};
specialRecon = '';

Res = 0.06;
cAngle = 80*pi/180;
x_range = [-4.1,4];
y_range = [-4,4.2];
z_range = [-4,4];

FILTER = '';
FILTERCUTOFF = [1 3]*1e6;
FILTERRFCOMP = 5;
ORDER = 6;

offset = 25;
INTERPOLATION = 0; % for PA interpolation
MULTISOS = 0;
SOS = 15;

DISPLAY = 1;
SAVE_DATA = 1;
DATBATCH = 1;
% TESTstart = 1;
% TESTend = 10;
DISPCAXIS = [-30 0];
FOURIER_FLAG = 0;
%%
if RTSAVING
    folder_names = dir(file_dir);
    folder_flags = [folder_names.isdir];
    folder_names = folder_names(folder_flags);
    folder_names = folder_names(3:end,:); file_names = folder_names;
    folder_num = size(folder_names,1);
    fprintf([num2str(folder_num),' folders detected in the directory.\n']);
    %     RFDataSize = [20000 256 1440];
    for ifolder = 1:folder_num
        currentfolder = [file_dir,folder_names(ifolder).name,'\'];
        criteriacheck = 0;    
        for icriteria = 1:length(FOLDERSELECTION)
            if contains(folder_names(ifolder).name,char(FOLDERSELECTION(icriteria)))
                criteriacheck = criteriacheck + 1;
            end
        end
        if criteriacheck < length(FOLDERSELECTION)
            continue;
        end
        parafile = dir(fullfile(currentfolder,'*.mat'));
        parafile = parafile.name;
        load([currentfolder,parafile]);
        disp(['Current folder: ',currentfolder]);
        datfiles = dir(fullfile(currentfolder,'*.dat'));
        
        BATCHSIZE = DATBATCH;
        DATNUM = size(dir(fullfile(currentfolder,'*.dat')),1);
        
        if rem(DATNUM,BATCHSIZE) == 0
            TESTstart = 1:BATCHSIZE:DATNUM;
            TESTend = BATCHSIZE:BATCHSIZE:DATNUM;
        else
            TESTstart = 1:BATCHSIZE:DATNUM;
            TESTend = BATCHSIZE:BATCHSIZE:DATNUM;
            TESTend = [TESTend DATNUM];
        end
        
        for idatfile = 2:length(TESTstart)%DATBATCH:size(datfiles,1)
            ibatchstart = TESTstart(idatfile);
            ibatchend = TESTend(idatfile);
            fprintf(['Current recon frames: ',num2str(ibatchstart),'-',...
                num2str(ibatchend), ' out of ', num2str(size(datfiles,1)),'\n']);
            batchiter = 1;
            
            for idatbatch = ibatchstart:ibatchend
                idatfilesname = datfiles(idatbatch).name;
                f = fopen([currentfolder,idatfilesname],'r');
                RFData0 = fread(f,'int16');
                RFData0 = reshape(RFData0,[length(RFData0)/256 RFSizeInfo(2) 1]);
                fclose(f);
                
                if BATCHTRANSFER
                    SATXNum = size(TX,2);
                    RFData1 = zeros(Receive(SATXNum).endSample,256,P.numFrames);
                    for iframe = 1:P.numFrames
                        RFData1(:,:,iframe) = ...
                            RFData0(Receive((iframe-1)*SATXNum+1).startSample:...
                            Receive(iframe*SATXNum).endSample,:);
                    end
                    
                    if idatbatch == ibatchstart
                        RFData = zeros(size(RFData1,1),size(RFData1,2),size(RFData1,3)*(ibatchend-ibatchstart+1));
                    end
                    
                    RFData(:,:,(batchiter-1)*size(RFData1,3)+1 : batchiter*size(RFData1,3)) = RFData1;
                    batchiter = batchiter + 1;
                end
            end
            %             file = [folder_names(ifolder).name,'....'];
            iteststart = 1;
            itestend = size(RFData,3);
            TXnum = size(TX,2);
            if FOURIER_FLAG
                Postprocess_2DUS_FFT(RFData,TXnum,Receive,10)
            end 
            
            if strcmp(specialRecon,'txwise')
                RFData0 = RFData;
                for isos = 1:length(SOS)
                    currTXNum = 1;
                    while currTXNum <= TXnum
                        temperature_c = SOS(isos);
                        run Recon_2DUS_Main
                        currTXNum = currTXNum + 1;
                        RFData = RFData0;
                    end
                    clearvars rTX rRCV angleAll
                end
            else
                if MULTISOS
                    RFData0 = RFData;
                    for isos = 1:length(SOS)
                        temperature_c = SOS(isos);
                        run Recon_2DUS_Main
                        clearvars rTX rRCV angleAll
                        RFData = RFData0;
                    end
                else
                    %                     temperature_c = 15;
                    recon_ts = tic;
                    clear RFData1 RFData0
                    run Recon_2DUS_Main
                    recon_te = toc(recon_ts);
                    fprintf(['Total Recon Time: ',num2str(recon_te),'s. \n'])
                end
            end
            
        end
        clearvars rTX rRCV angleAll
    end % finish loading file
else
    file_names = batchReader(file_dir,'*.dat',{'TXmode1_dist5'});
    for ifile = 1:size(file_names,1)
        TESTstart = 1;
        TESTend = 40;
        for iReconBatch = 1:length(TESTstart)
            file = strtrim([file_names(ifile,1:end)]);
            fprintf(['Loading file: ',file, ' \n'])
            if ~exist('P','var')
                parafile = [file(1:end-5),'1.mat'];
                load([file_dir,parafile]);
            end
            if P.numFrames < TESTstart(iReconBatch)
                break
            end
            f = fopen([file_dir,file],'r');
            RFData = fread(f,'int16');
            fclose(f);
            
            RFData = single(reshape(RFData, RFSizeInfo'));
            RFData = RFData(Receive(1).startSample:Receive(end).endSample,:,:);
            SATXNum = size(TX,2);
            if BATCHTRANSFER
                RFData1 = zeros(Receive(SATXNum).endSample,256,P.numFrames);
                for iframe = 1:P.numFrames
                    RFData1(:,:,iframe) = ...
                        RFData(Receive((iframe-1)*SATXNum+1).startSample:...
                        Receive(iframe*SATXNum).endSample,:);
                end
                RFData = RFData1;
                clear RFData1
            end
            
            iteststart = TESTstart(iReconBatch);
            itestend = TESTend(iReconBatch);
            P.acqStartDepth = P.startDepth;
            if MULTISOS
                RFData0 = RFData;
                for isos = 1:length(SOS)
                    temperature_c = SOS(isos);
                    run Recon_2DUS_Main
                    clearvars rTX rRCV angleAll
                    RFData = RFData0;
                end
            else
                clear RFData1 RFData0
                run Recon_2DUS_Main
            end 
        end
    end
end