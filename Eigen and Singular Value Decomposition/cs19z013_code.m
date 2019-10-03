%% Reading the image file and converting it from unit8 to double
matrixA = imread("46.jpg");
A=im2double(matrixA)
%% Eigen Value Decomposition
    [X,lamda] = eig(A);
    
    [~,I] = sort(diag(lamda),'descend'); %%sorting eigen values and corresponding vectors
     X= X(:, I);
     lamda=lamda(I,I);
invx=inv(X);
templamda=lamda;
i =1;
for r=[1,10,100,200,256] % selecting first 1,10,100,200,256 maximum values
lamda=templamda;
for c = r+1:256
        lamda(c,c)= 0;
end

 B=X*(lamda)*invx;
 difference=A-B;
 evd_error(r)=norm(difference,'fro'); %%norm to find out the error
 subplot(1,5,i);
 imshow(im2uint8(A-B));
 i=i+1;
end


ata=transpose(A)*A;
[v,s] = eig(ata);
[~,I] = sort(diag(s),'descend'); %%sorting singular values and corresponding vectors
v= v(:, I);
     s=s(I,I);

s=sqrt(s);
invs=inv(s);
u=A*v*invs;
temps=s;
 i=1;
  for r=[1,10,100,200,256] %%selecting first 1,10,100,200,256 maximum values
  s=temps;
  for c = r+1:256
        s(c, c) = 0;
  end
svdB= u*(s)*v';
svd_difference=A-svdB;
svd_error(r)=norm(svd_difference,'fro');
subplot(1,5,i);
imshow(im2uint8(svdB));
i=i+1;
  end

