#include "mex.h" /* Always include this */
#include "matrix.h" 
#include <stdlib.h>
#include <string.h>
/*

C code for
function [w,g] = cdescentCycle(X, g, w, rand_idx, shrinkFactor, threshold)
  
for j=rand_idx
    wjold = w(j);
    
    gj = g + w(j)*X(:,j);
    wj = sum(X(:,j).*gj);
    
    w(j) = sign(wj) * max((abs(wj) - threshold), 0) / shrinkFactor(j);
    
    g = g - X(:,j)*(w(j)-wjold);
end

 */

void  do_iter(double *w, double *g, /* output */
              const double *X, const mwSize Xrows, const mwSize Xcols, const double *g_orig, const double *w_orig,
              const int64_t *rand_idx, mwSize num_rand_idx, const double *shrinkfactor, const double threshold) /* input */
{

  memcpy((void *)w, (const void *)w_orig, (size_t)Xcols * sizeof(double));  
  memcpy((void *)g, (const void *)g_orig, (size_t)Xrows * sizeof(double));

  
  mwSize j_iter,j,i;  
  double wjold,wjnew;
  mxArray *aux[1];
  aux[0] = mxCreateDoubleMatrix(Xrows, 1, mxREAL);
  double *gj = (double *)mxGetPr(aux[0]);
  
  for (j_iter=0; j_iter<num_rand_idx; j_iter++) {
    j = rand_idx[j_iter] - 1; /* matlab index starts from 1  */ 
    /*if(j >= Xcols)
      mexErrMsgTxt("num index j exceeded the size of w");*/
      
    wjold = w[j];

    /* Regress j-th partial residuals on j-th predictor */
    for(i=0;i<Xrows;i++){        
      gj[i] = g[i] +  wjold * X[i + Xrows*j]; 
    }

    wjnew = 0.0;
    for(i=0;i<Xrows;i++){
      wjnew += X[i+Xrows*j] * gj[i];
    }

    /* Soft thresholding */
    if ( wjnew >= 0 ) 
      wjnew = ( wjnew-threshold > 0 ) ? (wjnew-threshold) / shrinkfactor[j] : 0 ; 
    else
      wjnew = (-wjnew-threshold > 0 ) ? (wjnew+threshold) / shrinkfactor[j] : 0 ;
      
    for(i=0;i<Xrows;i++){
      g[i] -= (wjnew - wjold ) * X[i + Xrows*j]; 
    }
    w[j] = wjnew;
  }
  mxDestroyArray(aux[0]); 

}


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{

  if( nlhs > 2 ) {
    mexErrMsgTxt("Too many outputs");
  }
  if( nrhs != 6 ) {
    mexErrMsgTxt("Need exactly 6 inputs");
  }
  /* [w,g] = (X, g, w, active,  shrinkFactor, threshold) */

  double *X;
  mwSize Xrows,Xcols;
  
  X= mxGetPr(prhs[0]);
  Xrows = mxGetM(prhs[0]);
  Xcols = mxGetN(prhs[0]);
  if( !mxIsDouble(prhs[0]) || mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]) ||
      mxGetNumberOfDimensions(prhs[0])!=2 ) {
    mexErrMsgTxt("X(1st argument) must be real double full matrix");
  }  

  double *g;
  double *w;
  g = mxGetPr(prhs[1]);
  if(!mxIsDouble(prhs[1]) || mxIsSparse(prhs[1]) || mxIsComplex(prhs[1]) || mxGetN(prhs[1]) != 1) {
    mexErrMsgTxt("g must be a real double column vector.");                      
  }
  if (Xrows != mxGetM(prhs[1])) {
    mexErrMsgTxt("Inner dimensions of matrix multiply do not match. (num row of X and dim of g "); 
/*    mexErrMsgTxt("Inner dimensions of matrix multiply do not match. (num row of X=%d and dim of g=%d ", Xrows, mxGetM(prhs[1])); */
  }
  
  w = mxGetPr(prhs[2]);
  if( !mxIsDouble(prhs[2]) || mxIsSparse(prhs[2]) || mxIsComplex(prhs[2]) ||  mxGetN(prhs[2]) != 1 ) {
    mexErrMsgTxt("w (3rd argument) must be real double column vector");
  }
  if (Xcols != mxGetM(prhs[2])) {
    mexPrintf("num col of X=%d and dim of w=%d ", Xcols, mxGetM(prhs[2])); 
    mexErrMsgTxt("Inner dimensions of matrix multiply do not match. (num col of X and dim of w ");
  }

  int64_t *rand_idx;
  mwSize num_rand_idx;
  double *shrinkfactor;
  double threshold;
  rand_idx = (int64_t *)mxGetData(prhs[3]);
  if( !mxIsInt64(prhs[3]) || mxIsSparse(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3]) != 1) {
    mexPrintf("size of rand_idx = %d %d ",  mxGetM(prhs[3]), mxGetN(prhs[3]));     
    mexErrMsgTxt("rand_idx must be a dense row int64 vector.");
  }
  num_rand_idx=mxGetN(prhs[3]);
    
  shrinkfactor = mxGetPr(prhs[4]);
  if( !mxIsDouble(prhs[4]) || mxIsSparse(prhs[4]) || mxIsComplex(prhs[4]) ){
    mexErrMsgTxt("shrinkFactor (5th argument) must be real double vector");
  }
  if (    !(1 == mxGetN(prhs[4]) && Xcols == mxGetM(prhs[4]) )
          && !(Xcols == mxGetN(prhs[4]) && 1 == mxGetM(prhs[4]) ) ){
      mexPrintf("num col of X=%d and dim of shrinkFactor=%d %d", Xcols, mxGetN(prhs[4]), mxGetM(prhs[4])); 
      mexErrMsgTxt("Inner dimensions of matrix multiply do not match. (num col of X and dim o f shrinkFactor ");
  }

  if( !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) ) 
    mexErrMsgTxt("thresholdis not real double");
  threshold = mxGetScalar(prhs[5]);
  

  double *retw;
  plhs[0] = mxCreateDoubleMatrix((mwSize)Xcols, 1, mxREAL);
  retw = mxGetPr(plhs[0]);

  double *retg;
  plhs[1] = mxCreateDoubleMatrix((mwSize)Xrows, 1, mxREAL);
  retg = mxGetPr(plhs[1]);

  do_iter(retw,retg, X, Xrows, Xcols, g, w, rand_idx, num_rand_idx, shrinkfactor, threshold);
  
  return;
}
