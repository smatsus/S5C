%% Test performance on cifar10 datasets. 

%% cd /X/application/.. from /X/application
cd ..

%% loading dataset

load ../data/cifar10.mat
L = 10;
A0 = labels+1;
Y = im2double(data);

%% run experiments

[CE, ET] = run_S5C(Y0,A0,L,20*L);

%% show clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
