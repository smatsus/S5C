## S5C

MATLAB implementation of [Selective Sampling-based Scalable Sparse Subspace Clustering](http://papers.nips.cc/paper/9408-selective-sampling-based-scalable-sparse-subspace-clustering.pdf) (NeurIPS '19). S5C algorithm selects subsamples based on the approximated subgradients and linearly scales with the number of data points in terms of time and memory requirements. It provides theoretical guarantees of the correctness of the solution.  

## Test run

 Mex file representation_learning/cdescentCycleC.mexa64 is built for 64-bit Linux. If running on other platform, first compile  representation_learning/cdescentCycleC.c to create mex file for your platform (see [Matlab documentation](https://www.mathworks.com/help/matlab/matlab_external/build-an-executable-mex-file.html)).

Examples how to run the code are given in run_examples/ directory. Example scripts are given for all datasets used in the paper.

## Datasets

Five datasets used in the paper ([MNIST](http://yann.lecun.com/exdb/mnist/), [Extended Yale B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html), [Hopkins155](http://www.vision.jhu.edu/data/hopkins155/), [Letter-rec](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition), and [COIL100](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)) can be found in the datasets directory.

CIFAR-10 and Devanagari are not included due to their size. CIFAR-10 can be downloaded from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html). Devanagari can be downloaded from [https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset).

## Citing

When using the code in your research work, please cite "Selective Sampling-based Scalable Sparse Subspace Clustering" by Shin Matsushima and Maria Brbic.

    @incollection{matsushima19_s5c,
    title={Selective Sampling-based Scalable Sparse Subspace Clustering},
    author={Matsushima, Shin and Brbi\'c, Maria},
    booktitle = {Advances in Neural Information Processing Systems 32},
    pages = {12416--12425},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/9408-selective-sampling-based-scalable-sparse-subspace-clustering.pdf}
    }

