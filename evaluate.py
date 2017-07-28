import os
from optparse import OptionParser

import pandas as pd
import numpy as np

import common
import cy_lda
import file_handling as fh


def main():
    usage = "%prog data_dir model_dir"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    data_dir = args[0]
    model_dir = args[1]

    print("Loading training vocabulary")
    vocab = fh.read_json(os.path.join(data_dir, 'train.vocab.json'))
    print("Loading test data")
    test_X, _, test_items = common.load_data(data_dir, 'test', vocab)
    n_test, _ = test_X.shape
    test_X_list = common.convert_data_to_list_of_lists(test_X)

    # create a model with an arbitrary number of topics
    model = cy_lda.LDA(K=1, V=len(vocab))
    # load the parameters from the model file
    model.load_parameters(os.path.join(model_dir, 'train.model.npz'))

    # fit the model to the test data and compute perplexity
    print("Evaluating on test data")
    perplexity = model.evaluate(test_X_list)
    print("Perplexity = %0.4f" % perplexity)

    # save the resulting document representations
    print("Saving document-topic matrix")
    gammas = model.get_document_representations()
    n_items, n_topics = gammas.shape
    document_topic_matrix = pd.DataFrame(gammas, index=test_items, columns=np.arange(n_topics))
    document_topic_matrix.to_csv(os.path.join(model_dir, 'test.document_topic_gammas.csv'))

    gammas_norm = gammas / np.sum(gammas, axis=1).reshape((len(test_X_list), 1))
    document_topic_matrix = pd.DataFrame(gammas_norm, index=test_items, columns=np.arange(n_topics))
    document_topic_matrix.to_csv(os.path.join(model_dir, 'test.document_topic_means.csv'))


if __name__ == '__main__':
    main()
