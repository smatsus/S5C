% Letter-rec

%% cd /X/application/.. from /X/application
cd ..

%% loading dataset
load ../data/letter-rec

A0 = gnd;
Y = fea;
L = length(unique(A0));

%% run_experiments

[CE, ET] = run_S5C(Y,A0,L,20*L);

%% show clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
