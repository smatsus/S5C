%
% Perform experiments on Yale B dataset.
%

%% cd /X/application/.. from /X/application
cd ..

%% loading dataset
load ../data/YaleBCrop025.mat             % We used the resized raw images provided along with the SSC codes.

L = 38;
p = 2016;
n = 64;

Y0 = Y;
N = n*L;

Y = [];
for i=1:L
  Y = [Y Y0(:,:,i)];
end

A0 = reshape(repmat(1:L,n,1),1,N);

%% run experiments

[CE, ET] = run_S5C(Y,A0,L,20*L);

%% show clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
