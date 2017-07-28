import os
from optparse import OptionParser

import pandas as pd
import numpy as np

import common
import cy_lda


def main():
    usage = "%prog input_dir output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-k', dest='k', default=20,
                      help='Number of latent factors: default=%default')
    parser.add_option('--n_init', dest='n_init', default=1,
                      help='Number of documents to use (per topic) for random initialization=%default')
    parser.add_option('--smoothing', dest='smoothing', default=1.0,
                      help='Smoothing for initialization=%default')
    parser.add_option('--alpha', dest='alpha', default=0.1,
                      help='Initial hyperparameter for document distributions: default=%default')
    parser.add_option('--tol', dest='tol', default=1e-4,
                      help='Tolerance for convergence (relative change in bound)=%default')
    parser.add_option('--max_epochs', dest='max_epochs', default=100,
                      help='Maximum number of passes through all data=%default')
    parser.add_option('--tol_inner', dest='tol_inner', default=1e-6,
                      help='Tolerance for convergence for per-document optimization=%default')
    parser.add_option('--max_inner_iter', dest='max_inner_iter', default=20,
                      help='Maximum number of iterations for per-document optimization=%default')
    parser.add_option('--display', action="store_true", dest="display", default=False,
                      help='Print topics after each epoch: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (must be > 0): default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    output_dir = args[1]
    input_prefix = 'train'

    k = int(options.k)
    alpha = float(options.alpha)
    n_init = int(options.n_init)
    smoothing = float(options.smoothing)
    tol = float(options.tol)
    max_epochs = int(options.max_epochs)
    tol_inner = float(options.tol_inner)
    max_inner_iter = int(options.max_inner_iter)
    display = options.display
    seed = options.seed
    if seed is not None:
        np.random.seed(seed)

    train_X, vocab, train_items = common.load_data(input_dir, input_prefix)

    model = cy_lda.LDA(K=k, V=len(vocab), alpha=alpha)

    print("Reshaping data")
    train_X_list = common.convert_data_to_list_of_lists(train_X)

    # fit the model
    print("Fitting the model")
    model.fit(train_X_list, tolerance=tol, max_epochs=max_epochs, inner_tol=tol_inner, max_inner_iterations=max_inner_iter, n_initial_docs=n_init, initial_smoothing=smoothing, vocab=vocab, display_topics=display)

    print("Saving the model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    betas = model.get_topics()
    document_word_df = pd.DataFrame(betas, index=vocab, columns=np.arange(k))
    document_word_df.to_csv(os.path.join(output_dir, input_prefix + '.topic_word_matrix.csv'))

    gammas = model.get_document_representations()
    document_topic_matrix = pd.DataFrame(gammas, index=train_items, columns=np.arange(k))
    document_topic_matrix.to_csv(os.path.join(output_dir, input_prefix + '.document_topic_gammas.csv'))

    gammas_norm = gammas / np.sum(gammas, axis=1).reshape((len(train_X_list), 1))
    document_topic_matrix = pd.DataFrame(gammas_norm, index=train_items, columns=np.arange(k))
    document_topic_matrix.to_csv(os.path.join(output_dir, input_prefix + '.document_topic_means.csv'))

    output_filename = os.path.join(output_dir, input_prefix + '.model.npz')
    model.save_parameters(output_filename)


if __name__ == '__main__':
    main()
