function [wCtr] = FGweights(str,f,idxImg,pixelList)

N = length(pixelList);
[H,W,~] = size(f);

%% Get Objectness Proposals
str2 = strcat(str,'.txt');
A = dlmread(str2);
X = A(2:end,:);

% Build Preliminary HeatMap
MAT = zeros(H,W);
spMAP = zeros(N,1);
wCtr = zeros(N,1);

for i = 1:100,
    hwin = X(i,5)-X(i,3)+1;
    wwin = X(i,4)-X(i,2)+1;
    wMat = gausswin(hwin,2)*gausswin(wwin,2)'; 
    for x = X(i,2):X(i,4),
        for y = X(i,3):X(i,5),
            spMAP(idxImg(y,x)) = spMAP(idxImg(y,x)) + (1001-1000)*wMat(y-X(i,3)+1,x-X(i,2)+1);
       end
    end
end

for i = 1:N,
    MAT(pixelList{i,1}) = spMAP(i);
end

ta = 1.5*sum(MAT(:))/(H*W);
I = MAT;
%% RGB to LAB
lab = vl_xyz2lab(vl_rgb2xyz(f));
tmpImg=reshape(lab,H*W,3);
meanCol=zeros(N,3);
for i = 1:N
    meanCol(i, :)=mean(tmpImg(pixelList{i},:), 1);
end  

for i = 1:H,
    for j = 1:W,
        f(i,j,:) = meanCol(idxImg(i,j),:);
    end
end

isFG = zeros(1,N);
for i = 1:N,
    if spMAP(i) > ta,
        isFG(i) = 1;
    end
end

%% Mean Pos
meanPos = zeros(N, 2);

for i = 1 : N
    [rows, cols] = ind2sub([H, W], pixelList{i});    
    meanPos(i,1) = mean(rows) / H;
    meanPos(i,2) = mean(cols) / W;
end

FGscore = zeros(1,N);
BGscore = zeros(1,N);

for i = 1:N,
    for j = 1:N,
        if i == j,
            continue;
        end
        if isFG(j) == 1,
            d = max((meanCol(i,:)-meanCol(j,:)).*(meanCol(i,:)-meanCol(j,:)));
            FGscore(i) = FGscore(i) + d;
        else
            BGscore(i) = BGscore(i) + max((meanCol(i,:)-meanCol(j,:)).*(meanCol(i,:)-meanCol(j,:)));
        end
    end
end

for i = 1:H,
    for j = 1:W,
        MAT(i,j) = BGscore(idxImg(i,j))/FGscore(idxImg(i,j));
    end
end

for i = 1:N,
    wCtr(i) = BGscore(i)/FGscore(i);
end

% MAT = mat2gray(MAT);
% ta = *sum(MAT(:))/(size(f,1)*size(f,2));
% figure,imshow(MAT.*(MAT>ta));
%imwrite((I.*(I>ta)),strcat(str,'_S.png'));

wCtr = (wCtr - min(wCtr)) / (max(wCtr) - min(wCtr) + eps);
thresh = graythresh(wCtr);  %automatic threshold
wCtr(wCtr < thresh) = 0;




