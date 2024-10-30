function Recon_2DPA_Drive_AD_gen_func(root_dir, in_file_dir, save_sub_dir, recon_params)
    arguments
        root_dir {mustBeText} = 'D:\Anthony\PACT\SparseArray\soft-array-deep-learning\src\';
        in_file_dir {mustBeText} = 'recon_code\data\raw_data\';
        save_sub_dir {mustBeText} = 'recon_code\data\raw_mat\'
        recon_params {mustBeA(recon_params, "struct")} = struct([])
    end
    %% Recon_PA_2D
    % clear,
    % clc, close all
    %g = gpuDevice(2);
    % reset(g);

    %%
    %root_dir = 'C:\Users\PI-Lab\Desktop\DOCUMENTS\Anthony\PACT\SparseArray\soft-array-deep-learning\src\';
    % root_dir = 'D:\Anthony\PACT\SparseArray\soft-array-deep-learning\src\';
    % file_dir = [root_dir, 'recon_code\data\raw_data\'];
    file_dir = [root_dir, in_file_dir];
    %save_sub_dir = 'recon_code\data\raw_mat\sensor_pos_err_quant\std_err_0-00000';
    save_dir = [root_dir, save_sub_dir];
    code_dir = './';
    params_dir = './+sparse_recon/params/';%'../recon_params/';%'./+sparse_recon/params/';
    
    %% Unpack Recon Params /  Flags into respective variables:

    % Equivalent to Unpacking kwargs in Python
    assignvars = structvars(recon_params);
    for i = 1:size(assignvars, 1)
        var_expr = assignvars(i, :);
        eval(var_expr)
    end

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
            criteriacheckpass = false;
            for icriteria = 1:length(FOLDERSELECTION)
                if contains(folder_names(ifolder).name,char(FOLDERSELECTION(icriteria)))
                    criteriacheckpass = true;
                end
            end
            if ~criteriacheckpass
                % Folder does not have any specified keyword so go to next one
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
                
                for iwl = RECON_WL
                    if MULTI_SOS
                        for isos = 1:length(SOS)
                            RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
                            clear angleAll idxAll
                            temperature_C = SOS(isos);
                            
                            fprintf(['Current wl: ',num2str(iwl),'/',...
                                num2str(MULTI_WL),'\n','Current tC: ',num2str(temperature_C),'\n']);
                            run sparse_recon/Recon_2DPA_Main_AD_sparse_recon_gen
                        end
                    else
                        RFData = RFData0(Receive(iwl+1).startSample:Receive(iwl+1).endSample,:,:);
    %                     clear angleAll idxAll
                        
                        fprintf(['Current wl: ',num2str(iwl),'/',...
                            num2str(MULTI_WL),'\n']);
                        run sparse_recon/Recon_2DPA_Main_AD_sparse_recon_gen
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

%% Subfunction Helper:

function varargout=structvars(varargin)
    %STRUCTVARS - print a set of assignment commands that, if executed, would 
    %assign fields of a structure to individual variables of the same name (or vice
    %versa).
    %
    %Examples: Given structure myStruct, with fields a,b,c, & d
    %
    % (1) structvars(myStruct)   %assign fields to variables
    % 
    %       
    %         a = myStruct.a;     
    %         b = myStruct.b;     
    %         c = myStruct.c;     
    %         d = myStruct.d;     
    % 
    % (2) structvars(3,myStruct)   %split the last result across 3 columns
    % 
    %         a = myStruct.a;     c = myStruct.c;     d = myStruct.d;     
    %         b = myStruct.b;                                             
    % 
    % (3) structvars(3,myStruct,0)  %assign variables to fields 
    % 
    %         myStruct.a = a;    myStruct.c = c;    myStruct.d = d;    
    %         myStruct.b = b; 
    %
    %The routine is useful when you want to pass many arguments to a function
    %by packing them in a structure. The commands produced by structvars(...)
    %can be conveniently copy/pasted into the file editor at the location in the file
    %where the variables need to be unpacked.
    %
    %
    %SYNTAX I:
    %
    %     assigns=structvars(InputStructure,RHS)
    % 
    %     in:
    %         InputStructure: A structure
    %         RHS: Boolean. If true (default), dot indexing expressions will be on the 
    %             right hand side.
    % 
    %     out:
    % 
    %      assigns: a text string containing the commands (see Examples above). 
    %               If omitted, the result will simply be displayed on the
    %               screen.
    % 
    % 
    %        NOTE: If the name of the variable passed as InputStructure cannot be 
    %               determined via inputname() a default name of 'S' will be used.
    %
    %
    %SYNTAX II: 
    %
    %  assigns=structvars(nCols,...)
    %
    %Same as syntax I, but assignment strings will be split across nCols
    %columns.
    %
    %
    % by Matt Jacobson
    %
    % Copyright, Xoran Technologies, Inc. 2009
    %
    if isnumeric(varargin{1}), 
        nCols=varargin{1};
        varargin(1)=[];
        idx=2;
    else
        nCols=1;
        idx=1;
    end
        
    nn=length(varargin);
    S=varargin{1};
    if nn<2, RHS=true; else RHS=varargin{2}; end
        
    fields=fieldnames(S);
    sname=inputname(idx); 
    if isempty(sname), sname='S'; end
    if RHS  
        assigns= cellfun(@(f) [f ' = ' sname '.' f ';     '],fields,'uniformoutput',0);
    else %LHS
        assigns= cellfun(@(f) [ sname '.' f ' = ' f ';    '],fields,'uniformoutput',0);    
    end
    L0=length(assigns);
    L=ceil(L0/nCols)*nCols;
    Template=false(nCols,L/nCols);
    Template(1:L0)=true;
    Template=Template.';
    Table=cell(size(Template));
    Table(:)={' '};
    Table(Template)=assigns;
    TextCols=cell(1,nCols);
    for ii=1:nCols
        TextCols{ii}=char(Table(:,ii));
    end
    assigns=[TextCols{:}];
    if nargout
        varargout={assigns}; 
    else
        disp ' '
        disp(assigns),
        disp ' '
    end
end