% Compares performance on devanagari

%% cd /X/application/.. from /X/application
cd ..

%% load dataset
L = 46;

Y = csvread('../data/devanagari_data.csv');
Y = Y';

load ../data/devanagari_label.mat;

A0 = labels;

%% run_experiments

[CE, ET] = run_S5C(Y0,A0,L,20*L);

%% show clustering error and elapsed time

CE
ET

%% cd /X/application from /X/application/..
cd application
