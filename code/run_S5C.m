% Run S5C algorithm. Lambda is cross-validated in range [2^-1, 2^-2,..., 2^-10].
%
% Param:
%       Y: data matrix (each column is a data point)
%       A0: labels of each datapoints
%       L: number of labels
%       numS: number of subsamples 
%       
% Return:
%       clustering_errors: clustering errors for each lambda
%       elapsed_times: elapsed times for each lambda
%
%
%  [Cite] Shin Matsushima and Maria Brbic,
%         "Selective Sampling-based Scalable Sparse Subspace Clustering"
%         Conference on Neural Information Processing Systems, 2019
%
%  version 1.0 -- Oct/2019
%
%  Shin Matsushima and Maria Brbic


function [clustering_errors, elapsed_times] = run_S5C(Y,A0,L,numS)

  %% addpath
  addpath representation_learning
  addpath spectral_clustering
  addpath utils

  %% for reproducible results
  s = RandStream('mcg16807','Seed',25);
  RandStream.setGlobalStream(s);
  initState = RandStream.getGlobalStream();

  %% Our S5C algorithm
  fprintf('Running S5C..\n'); 

  elapsed_times = [];	 
  clustering_errors = [];
  iter = 1;  
  for plambda = 1:10
    stream.State = initState;
    lambda = 2^-plambda
	
    start1 = tic;
    [C,~] = representation_learning_S5C(normc(Y), lambda, numS);
    elapsed_time_1 = toc(start1);

    W = abs(C) + abs(C)';
	
    start2 = tic;
    A = spectral_clustering_S5C(W, L);
    elapsed_time_2 = toc(start2);
    
    elapsed_times(iter) = elapsed_time_1 + elapsed_time_2;
    clustering_errors(iter) = clustering_error(A,A0);      
    iter = iter + 1;	
  end    
end
