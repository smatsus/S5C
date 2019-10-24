% Performs spectral clustering using orthogonal iteration
% With power iteration based methods we are finding k largest eigenvalues
% instead of smallest, so we need to use matrix
% lambda_max*I-L_sym, where lambda_max is maximum
% eigenvalue of L_sym, and L_sym is normalized Laplacian matrix defined
% as I-D^{-1/2}WD^{-1/2}.
%
% Param:
%       W: affinity matrix (NxN symmetric matrix)
%       K: number clusters
%       err_tol: error tolerance for power deflation and orthogonal
%                iteration (default: 1e-3)
%       max_iter: maximum number of iteration for power deflation and
%                 orthogonal iteration (default 100)
%       verbose: verbosity 
%       
% Return:
%       A: cluster assignment of data points
%
%  [Cite] Shin Matsushima and Maria Brbic,
%         "Selective Sampling-based Scalable Sparse Subspace Clustering"
%         Conference on Neural Information Processing Systems, 2019
%
%  version 1.0 -- Oct/2019
%
%  Shin Matsushima and Maria Brbic


function [ A ] = spectral_clustering_S5C(W, K, err_tol, max_iter, verbose)

  % default parameters
  if ~exist('err_tol','var')
    err_tol = 1e-5;
  end
  if ~exist('max_iter','var')
    max_iter = 100;
  end
  if ~exist('verbose','var')
    verbose = false;
  end

  %%
  spectral_clustering_tic = tic;

  N = size(W, 1);

  isolated = (sum(W) ==0);
  %num_isolated = full(sum(isolated));
  connected = ~isolated;
  W = W(connected,connected);
  Nold = N;
  N = size(W,1);
  
  D = sparse(1:N,1:N, sum(W));
  L = D-W; % unnormalized Laplacian
  
  DN = sparse(1:N,1:N,1./sqrt(sum(W)));
  Lsym = speye(N) - DN*W*DN; % normalized Laplacian
  
  lambda_max = 2;
  
  % eigenvectors corresponding to smallest eigenvalues of L_sym
  % are eigenvectors corresponding to largest eigenvalues of L_max
  
  Lmax = lambda_max*speye(N) - Lsym;
  
  U = [];  
  Ucrr = orth_iter(Lmax, K, err_tol, max_iter);
  
  %normalize U
  for i = 1:N
    Ucrr(i,:) = Ucrr(i,:) ./ norm(Ucrr(i,:));
  end
  
  U = Ucrr;
  N = Nold;
  spectral_clustering_time = toc(spectral_clustering_tic);
  
  rng(1234)

  kmeanstic = tic;
  A = litekmeans(U, K, 'MaxIter',200, 'Replicates',20); 
  if(size(A,1) * size(A,2) ~= N)
    tmp = randi(K,[N,1]);
    tmp(connected) = A;
    A = tmp;
  end
  kmeans_time = toc(kmeanstic);

  if verbose
    timestats = containers.Map();
    timestats('1. spectral') = spectral_clustering_time;
    timestats('2. kmeans') = kmeans_time;
    timestats = [keys(timestats) ; values(timestats)]     
  end

