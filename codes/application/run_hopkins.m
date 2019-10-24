%% run on all datasets in hopkins 155 

%% cd /X/application/.. from /X/application
cd ..

%% load dataset
load ../data/Hopkins155_titles.mat;


%% run experiments for all 155 datasets
CE = [];
ET = [];
i_data = 1;

for i_data=1:length(Hopkins155_titles)
  i_algo=1;
  fprintf('%d/%d %s \n',i,length(Hopkins155_titles),Hopkins155_titles{i_data});
  eval(['load ../data/Hopkins155/' Hopkins155_titles{i_data} '/' Hopkins155_titles{i_data} '_truth.mat' ]);
  
  L = max(s);
  
  N = size(x,2);
  F = size(x,3);
  p = 2*F;
  Y = reshape(permute(x(1:2,:,:),[1 3 2]),p,N);
  
  [CE_one, ET_one] = run_S5C(Y0,A0,L,20*L);

  CE = [CE CE_one];
  ET = [ET ET_one];
 
end

%% display results
% all
CE
ET
% mean
mean(CE)
mean(ET)

%% cd /X/application from /X/application/..
cd application
