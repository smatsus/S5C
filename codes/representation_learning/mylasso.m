function STATS = mylasso(X,S,STATS, i)
%myLASSO Perform lasso with warm start.
%    STATS = mylasso(X,Y,S,STATS,i) performs 
%    minimize 0.5 * || Y - Xw ||_2^2 + Lambda * ||w||_1
%    subject to  w_j = 0   j notin S.
%
%    where w is i-th column of STATS.C.
%    
%
%   we perform cyclic coordinate descent with initial value of w,
%   with the side information r = Xw - Y. r is stored as i-th column
%   of STATS.R.
%
%   Positional parameters:
%
%     X                A numeric matrix (dimension, say, MxN)
%     Y                A numeric vector of length M
%     S                An index set that represents S 
%     STATS            STATS is a struct that contains information about 
%                      optimization. STATS contains the following fields: 
%
%       'C'            Initial value of w (coefficients) is stored
%                      in C(:,i) as a sparse vector. 
%       'R'            r = Xw - Y, where w is given initial value
%                      of coefficients, is srored in R(:,i) as a
%                      dense vector.
%
%       'Lambda'       The Value of lambda.
%       'reltol'       Convergence threshold for coordinate descent algorithm.
%                      The coordinate descent iterations will terminate
%                      when the relative change in the size of the
%                      estimated coefficients B drops below this threshold.
%                      Default: 1e-4. Legal range is (0,1).
%   
%   
%   Return values:
%     STATS           The same struct as input argument.
%                      
%                      


% --------------------
% Lasso model fits
% --------------------
norms = STATS.normsSt;
XS = STATS.XSt;

w_size = size(S,2);
w_vec = STATS.W(1:w_size,i);
% w_vec is a dense column vector of
% sum(S==1) by 1

fittic =tic;
[~,diag_col] = find(S==i);
[w_vec,r,time_for_CD] = lassoFit(diag_col, XS, w_vec, STATS.R(:,i), ...
                                 STATS.Lambda, ...
                                 STATS.reltol, norms);
time_for_fit = toc(fittic);

STATS.R(:,i) = r;
STATS.W(1:w_size,i) = w_vec;

STATS.time_for_CD = STATS.time_for_CD + time_for_CD ;
STATS.time_for_fit = STATS.time_for_fit + time_for_fit ;

end



% ===================================================
%                 lassoFit() 
% ===================================================
function [w, r, time_for_CD] = lassoFit(diag_col, X, ...
                                        w, r, ...
                                        threshold, reltol, norms)
                                              
% [N,P] = size(X);
%%% Note that in the context of SSC,
%%% P is number of datapoints and  
%%% N is number of features

t_other = zeros(1,3);

active = (w ~= 0)';

time_for_CD = 0;

wold = w;
 
% Iterative coordinate descent until converged
while true

  start_CD = tic;

  rand_idx = int64(setdiff(find(active), diag_col) );
  [w,r] = cdescentCycle(X,r,w,rand_idx, active,norms,threshold);
  active = (w ~= 0)';
 
  time_for_CD = time_for_CD + toc(start_CD);

  if norm( (w-wold) ./ (1.0 + abs(wold)), Inf ) < reltol
    % Cycling over the active set converged.
    % Do one full pass through the predictors.
    % If there is no predictor added to the active set, we're done.
    % Otherwise, resume the coordinate descent iterations.
    wold = w;
    
    potentially_active = abs(r' *X) > threshold;
     
    if any(potentially_active)
      new_active = active | potentially_active;
       
      start_CD = tic;

      rand_idx = int64(setdiff(find(new_active), diag_col) );
      [w,r] = cdescentCycle(X,r,w,rand_idx,new_active,norms,threshold);
      new_active = (w ~= 0)';
      
      time_for_one_CD = toc(start_CD);
      time_for_CD = time_for_CD + time_for_one_CD ;
    else
      new_active = active;
    end

    if isequal(new_active, active)
      break
    else
      active = new_active;
    end
    
    if norm( (w-wold) ./ (1.0 + abs(wold)), Inf ) < reltol
      break
    end
    
  end % of if norm( (w-wold) ./ (1.0 + abs(wold)), Inf ) < reltol

  wold = w;
    
end % of while true

end

% ===================================================
%                 cdescentCycle() 
% ===================================================

function [w,r] = cdescentCycle(X,r,w, rand_idx,active,norms, ...
                               threshold)

num_rand = size(rand_idx,1) * size(rand_idx,2);
if(num_rand == 0)
  return;
else
  rand_idx = rand_idx(randperm(num_rand));
end

[w,r] = cdescentCycleC(X,r,w,rand_idx,norms,threshold);
 
end %-cdescentCycle

