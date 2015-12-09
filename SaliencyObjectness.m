function [ wCtr ] = SaliencyObjectness(bbox_filename,H,W,idxImg,pixelList,adjcMatrix,colDistM,clipVal)
% SAI SRIVATSA R
% Email: saisrivatsan12@gmail.com
% Date: 12/09/2015


N = length(pixelList);
% Set the number of proposals to consider
% For fast mode, set nProposals to 200. For more accurate results but 
% at a slower rate, set nProposals to 1000.
nProposals = 1000;   

% Reads Objectness Proposals
A = dlmread([ 'BingBoxes/' bbox_filename '.txt']);

X = A(2:end,:);
MAT = zeros(H,W);
spMAP = zeros(N,1);

for i = 1:nProposals,
    hwin = X(i,5)-X(i,3)+1;
    wwin = X(i,4)-X(i,2)+1;
    wMat = gausswin(hwin,2)*gausswin(wwin,2)'; 
    for x = X(i,2):X(i,4),
        for y = X(i,3):X(i,5),
            spMAP(idxImg(y,x)) = spMAP(idxImg(y,x)) + X(i,1)*wMat(y-X(i,3)+1,x-X(i,2)+1);
       end
    end   
end

for i = 1:N,
    MAT(pixelList{i,1}) = spMAP(i);
end

% Objectness Threshold Map
ta = 1.5*sum(MAT(:))/(H*W);
isFG = spMAP>ta;

geoDistMatrix = CalGeoDist(adjcMatrix, colDistM, clipVal);

FGscore = zeros(N,1);
BGscore = zeros(N,1);

for i = 1:N,
    for j = 1:N,
        if isFG(j) == 1,
            FGscore(i) = FGscore(i) + geoDistMatrix(i,j);
        else
            BGscore(i) = BGscore(i) + geoDistMatrix(i,j);
        end
    end
end

wCtr = zeros(N,1);
for i = 1:N,
    MAT(pixelList{i,1}) = BGscore(i)/FGscore(i);
    wCtr(i) = BGscore(i)/FGscore(i);
end

wCtr = (wCtr - min(wCtr)) / (max(wCtr) - min(wCtr) + eps);
thresh = graythresh(wCtr);  %automatic threshold
wCtr(wCtr < thresh) = 0;

end

