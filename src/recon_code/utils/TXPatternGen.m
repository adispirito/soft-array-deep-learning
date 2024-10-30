function e1 = TXPatternGen(x_trans,y_trans,minDist)
% minDist in mm: 5 mm ~ 50 TX
% Configure 1
% In a set of coordinates, find N points such that their minimum distance
% is maximized
% para_path='H:\Dropbox\0Code\2D\SetupScripts\Params\';
% x_trans = load([para_path,'x_trans.mat']);
% x_trans = x_trans.x_trans;
% % load([para_path,'z_trans.mat']);
% y_trans = load([para_path,'y_trans.mat']);
% y_trans = y_trans.y_trans;
% load([para_path,'ele_trans.mat']);
% load([para_path,'azi_trans.mat']);
%% Distmap and elemloc
% elemDist = zeros(length(x_trans));
% for ielem = 1:length(x_trans)
%     elemDist(ielem,:) = sqrt(x_trans(ielem)^2 + y_trans.^2);
% end
% figure,imagesc(1:length(x_trans),1:length(y_trans),elemDist)
% pbaspect([1 1 1]),xlabel('Elem #.'),ylabel('Elem #.')
% h = colorbar; ylabel(h,'Distance [mm]')
%
% figure,scatter(x_trans,y_trans)
% pbaspect([1 1 1]),xlabel('[mm]'),ylabel('[mm]')

%% Elem numbering
ElemRingL = [1 8;9 18;19 31;32 46;47 63;64 82;83 104;105 128];
ElemRingR = [129 135;136 145;146 157;158 172;173 190;191 210;211 232;233 256];

%% Choose start elem in ring 1 and define min dist
e0 = [1 6 131];
% minDist = 5; % [mm]
e1 = e0;

for iring = 2:8
%     iring
    ielemring = [ElemRingL(iring,1):ElemRingL(iring,2) ElemRingR(iring,1):ElemRingR(iring,2)];
    idist1 = zeros(length(e1),length(ielemring));
    for ire = 1:length(e1)
        idist1(ire,:) = sqrt((x_trans(ielemring)-x_trans(e1(ire))).^2 + (y_trans(ielemring)-y_trans(e1(ire))).^2);
    end
    idist2 = idist1 > minDist;
    idist3 = sum(idist2,1);
    ind1 = find(idist3>=length(e1));
    
    idist4 = zeros(length(ind1));
    for jre = 1:length(ind1)
        idist4(jre,:) = sqrt((x_trans(ind1(jre))-x_trans(ind1)).^2 + ...
            (y_trans(ind1(jre))-y_trans(ind1)).^2);
    end
    
    if length(ind1)>1
        idist5 = idist4 > minDist;
        iter = 1;
        temp = 1;
        ind2 = [1];
        while iter <= length(ind1)
            temp1 = find(idist5(temp,:)>0);
            for iind = 1:length(temp1)
                j = sum(idist5(temp1(iind),ind2));
                if j == length(ind2)
                    ind2 = [ind2 temp1(iind)];
                    break;
                end
            end
            iter = iter + 1;
            temp = ind2(end);
            
        end
        
        %     idist6 = sum(idist5,1);
        %     [~,ind2] = max(idist6);
        %     ind3 = ind1(idist5(ind2,:));
        e1 = [e1 ielemring(ind1(ind2))];
    end
%     figure(1),scatter(x_trans,y_trans),hold on
%     scatter(x_trans(e1),y_trans(e1)),hold off
%     pbaspect([1 1 1])
%     title(['run ',num2str(iring)])
end
%%
% figure(2),scatter(x_trans,y_trans),hold on
% scatter(x_trans(e1),y_trans(e1)),hold off
% pbaspect([1 1 1])
% title(['TX num = ',num2str(length(e1))])
% legend('All Elem','TX Elem')

end


