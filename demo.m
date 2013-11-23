clear;
clc;

rng(1943);

D = 8;
N = 4000;
numTest = 100;
numBases = 100;

lambda = 0.1;
sigma = 2^-0.5;

trainX = (0.08)*randn(D,N);
testX = (0.08)*randn(D,numTest);
w = randn(1,D);
trainY = w*trainX;
testY = w*testX;

% linear
tic
w_hat = trainY*trainX'/(trainX*trainX'+lambda*eye(D));
y_hat = w_hat*testX;
fprintf('Linear Model (Ridge Regression) MSE: %f \n', mean((testY-y_hat).^2));
toc
% exact kernel
tic
Ktrain = rbf(trainX,trainX,sigma);
alpha = trainY / (Ktrain+lambda*eye(N));
Ktest = rbf(trainX,testX,sigma);
y_hat = alpha*Ktest;
fprintf('Kernel Regression (RBF + L2) MSE: %f \n', mean((testY-y_hat).^2));
toc
% random kitchen sinks
tic
W = sigma^-1*randn(numBases,D);
trainX_proj = numBases^-0.5*exp(sqrt(-1)*W*trainX);
w_hat = symmlq(@(v)(lambda*v(:) + trainX_proj*(trainX_proj'*v(:))),trainX_proj*trainY(:),1e-6,2000);
testX_proj = numBases^-0.5*exp(sqrt(-1)*W*testX);
y_hat = real(w_hat'*testX_proj);
fprintf('Kernel Regression (Ramdom Kitchen Sinks + RBF + L2) MSE: %f \n', mean((testY-y_hat).^2));
toc
% compare two kernels
figure;
%colormap(gray);
subplot(2,2,1);
approxK = real(trainX_proj'*trainX_proj);
imagesc(approxK);
title('Approx. K');
colorbar;
subplot(2,2,2);
imagesc(Ktrain);
title('Exact K');
colorbar;
subplot(2,2,3);
imagesc(Ktrain-approxK);
title('Diff.');
colorbar;
subplot(2,2,4)
imagesc(abs(Ktrain-approxK));
title('Abs. Diff.');
colorbar;