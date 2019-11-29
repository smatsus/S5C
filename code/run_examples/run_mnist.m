% Compares performance on mnist dataset.

%% cd /X/application/.. from /X/application
cd ..

%% load dataset
load ../data/mnist_all;
L = 10;    
Y0 = images;
A0 = labels + 1; % labels start from 1

%% run_experiments

[CE, ET] = run_S5C(Y0,A0,L,20*L);

%% show clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
