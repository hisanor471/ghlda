The code is written using c++/pybind11.
See usage_*.ipynb to see how to run the algorithm.
Data is located in the data folder. You need to additionally download the pre-trained word embedding vectors, as described in the text [1].
The results folder is a placeholder to store estimated results.
Last but not least, make sure you set path_home as the same path as to where this README file is located.

Dependencies:
(1) c++
gsl (2.3)
Eigen (3.3.5)
OpenMP (4.5)
pybind11 (2.4.3)

(2) python
pandas (0.25.3)
numpy (1.17.3)
gensim (3.8.1)

[1] glove https://nlp.stanford.edu/projects/glove/
word2vec https://code.google.com/archive/p/word2vec/
fasttext https://fasttext.cc/docs/en/english-vectors.html
