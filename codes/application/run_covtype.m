% Forest cover type dataset from UCI Machine Learning repository
%% Loading dataset

%% cd /X/application/.. from /X/application
cd ..

load ../data/covtype.mat             % We used the resized raw images provided along with the SSC codes.
L = 7;

%% Run experiments

[CE, ET] = run_S5C(Y0,A0,L,20*L);

%% Display clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
