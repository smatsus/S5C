% Orthogonal iteration method for the approximation of the k eigenvectors
% associated with the largest eigenvalues. Orthogonal Iteration is a block
% version of the Power Method where vectors are forced to be orthogonal to
% each other.
% (http://mlwiki.org/index.php/Power_Iteration).
%
% Param:
%       A: affinity matrix (NxN symmetric matrix)
%       K: number of eigenvectors to approximate
%       err_tol: error tolerance for power method
%       max_iter: maximum number of iterations for power method
% Return:
%       Q: Nxk matrix of eigenvectors
%
function [ Q ] = orth_iter(  A, K, err_tol, max_iter, seed )

if ~exist('seed','var')
    seed = 1;
end
rng(seed);

N = size(A,1);

Q = rand(N,K);
[Q,~] = qr(Q,0);

Q_prev = Q;

for i = 1:max_iter
    Z = A*Q;
    [Q,~] = qr(Z,0);
    
    %norm(Q-Q_prev)
    
    if norm(Q-Q_prev, 'fro') / sqrt(N*K) < err_tol
        break
    end
    Q_prev = Q;
end

end

