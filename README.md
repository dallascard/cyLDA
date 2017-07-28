cyLDA
====

### Overview

This is a Cython implementation of David Blei's variational inference code for LDA, available at https://github.com/blei-lab/lda-c

Note: My primary purpose here was to learn about Cython. Many more parts could be cythonized, but the slowest parts have been dealt with, and it is already competitive in terms of speed with the original C implementation. However, there is little reason to use this implementation compared to say, [Mallet](http://mallet.cs.umass.edu/). Nevertheless, feel free to borrow or extend this code as you wish.

### Requirements

- python3
- Cython
- numpy
- scipy
- pandas
- spaCy


### Data format

This implementation expects three files for input:

- train.items.json: a json file containing a list of D document IDs 
- train.vocab.json: a json file containing a list of V words
- train.npz: a sparse DxV matrix of word counts saved as a npz file in scipy sparse COO format
  
  
`preprocess_data.py` will produce these files if given two json files (train and test), each of which should be a dictionary of documents, with each document's text in a field called 'text'.
 
 As an example, first run `download_20ng.py` to download the benchmark 20 newsgroups dataset as sets of train and test files, and then run: `preprocess_data.py data/20ng/20ng_all/train.json data/20ng/20ng_all/test.json data/20ng/20ng_all`
 

### Usage

Before fitting a model, it is necessary to compile the Cython code.

First, run:
`python setup.py build_ext --inplace`
which should compile cy_lda.pyx and util.pyx.

To fit an LDA model to data, run
`python fit.py input_dir model_dir`
 
For example: `python fit.py data/20ng/20ng_all data/20ng/20ng_all/model`

To display the top words for each topic after each iteration add the keyword `--display`. Use `-h` for more options.

To evalute the perplexity on held out data (approximated using the variational lower bound), run
`python evalute.py data_dir model_dir`

For example, `python fit.py data/20ng/20ng_all data/20ng/20ng_all/model`

### References

Most of the variable names in this code match those used in the original LDA paper:

-  David M. Blei , Andrew Y. Ng , Michael I. Jordan. *Latent Dirichlet Allocation*. 	Journal of Machine Learning Research. (2002)
