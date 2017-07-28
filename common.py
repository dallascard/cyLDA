import os
import numpy as np

import file_handling as fh


def load_data(input_dir, input_prefix, vocab=None):
    """
    Load data from three files: a .npz file with a DxV sparse matrix of word counts, a .vocab.json file with 
    the corresponding list of V words, and a .items.json file with a list of D document IDs
    :param input_dir: The directory contining the three files
    :param input_prefix: The prefix of all three files (e.g. train)
    :param vocab: optionally, a vocab from another dataset (for when loading test data)
    :return: (a DxV scipy.sparse.
    """
    print("Loading data")
    items = fh.read_json(os.path.join(input_dir, input_prefix + '.items.json'))
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    X = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz'))
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    return X, vocab, items


def convert_data_to_list_of_lists(X):
    """
    Convert a sparse matrix of counts to a list of lists of (word index, word count) tuples
    :param X: a DxV sparse matrix of word counts
    :return: a list of D lists, each containing the (word index, word count) tuples for one document
    """
    n, p = X.shape
    X_list = []
    X = X.tocsr()
    for i in range(n):
        nonzero_indices = X[i, :].nonzero()[1]
        nonzero_counts = np.array(X[i, nonzero_indices].todense(), dtype=int).reshape((len(nonzero_indices),))
        tuples = list(zip(nonzero_indices, nonzero_counts))
        X_list.append(tuples)

    return X_list