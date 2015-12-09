function [ Bscore ] = BGweights(f,idxImg,pixelList,colDistM,posDistM)

N = length(pixelList);
isBorder = zeros(N,1);
[H,W,~] = size(f);
isBorder(idxImg(1:H,1)) = 1;
isBorder(idxImg(1:H,W)) = 1;
isBorder(idxImg(1,1:W)) = 1;
isBorder(idxImg(H,1:W)) = 1;

Bscore = zeros(N,1);

for i = 1:N,
    for j = 1:N,   
        if isBorder(j)==0
            continue;
        end
        Bscore(i) = Bscore(i) + colDistM(i,j)*posDistM(i,j);
    end
end

Bscore = (Bscore-min(Bscore))/(max(Bscore)-min(Bscore));
Bscore = 1-Bscore;

MAT = zeros(H,W);
for i = 1:N,
    MAT(pixelList{i}) = Bscore(i);
end

% figure,imshow(mat2gray(MAT));

end

