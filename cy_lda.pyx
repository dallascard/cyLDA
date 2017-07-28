import math
import numpy as np
from scipy.special import gammaln   # log(gamma(x))
from scipy.special import psi       # aka digamma(x) aka polygamma(0, x)
from scipy.special import polygamma
from scipy.misc import logsumexp
from libc.math cimport exp
from libc.math cimport log

cimport cython
from util cimport log_sum_exp_vec, digamma, log_Gamma

class LDA:
    """
    Vanilla LDA optimized with variational EM, treating topics as parameters, with scalar smoothing parameter

    Generative model:
    \theta_d ~ Dirichlet(\alpha) for d = 1 .. D
    z_{dn} ~ Multinomial(\theta_d) for d = 1 .. D, n = 1 .. N_d
    w_{dn} ~ Multinomial(\beta_{z_{dn}}) for d = 1 .. D, n = 1 .. N_d

    Input a matrix of D x V word counts (X)
    """

    def __init__(self, K, V, alpha=0.1):
        """
        Create an LDA model
        :param K: The number of topics (int)
        :param V: The size of the vocabulary
        :param alpha: Initial hyperparameter for document-topic distributions
        """
        self.K = K                      # scalar number of topics
        self.V = V                      # scalar size of vocabulary
        self.D = None                   # scalar number of documents
        self.gammas = None              # D x K matrix of gammas
        self.log_betas = None           # V x K matrix of log(\beta)
        self.alpha = alpha              # scalar initial hyperparameter for p(\theta)
        self.bound = 0                  # the variational bound
        self.sum_d__ll_per_token = 0

    def fit(self, X, tolerance=1e-4, min_epochs=2, max_epochs=100, initial_smoothing=1.0, n_initial_docs=1, sequential_init=False, max_inner_iterations=20, inner_tol=1e-6, vocab=None, display_topics=False):
        """
        fit a model to data
        :param X: a list of documents (each a list of (word_index, count) tuples)
        :param tolerance: stopping criteria (relative change in bound)
        :param min_epochs: the minimum number of epochs 
        :param max_epochs: the maximum number of epochs
        :param initial_smoothing: smoothing to use when initializing topics
        :param n_initial_docs: the number of documents to use to randomly initialize each topic
        :param sequential_init: initialize topics using the first n_initial_docs documents in order
        :param max_inner_iterations: maximum number of iterations for inner optimization loop (E-step)
        :param inner_tol: the tolerance for the inner optimization loop (E-step)
        :param vocab: a list of words in the vocabulary (used for displaying topics)
        :param display_topics: if True, print topics after each epoch
        :return: None
        """
        # set up initial values for monitoring convergence
        prev_bound = -np.inf
        delta = np.inf
        self.D = len(X)

        # initialize model parameters based on data
        self.init_parameters(X, initial_smoothing, n_initial_docs=n_initial_docs)

        # repeat optimization until convergence
        print("Iter bound\tLL/token\tDelta")
        for i in range(max_epochs):
            # update parameters (this also computes the bound)
            self.update_parameters(X, max_inner_iterations=max_inner_iterations, inner_tol=inner_tol)

            # compute the relative change in the bound
            if i > 0:
                delta = (prev_bound - self.bound) / float(prev_bound)

            # print progress
            print('%d\t%0.3f\t%0.3f\t%0.5f' % (i, self.bound, self.sum_d__ll_per_token / self.D, delta))

            # store the new value of the bound
            prev_bound = self.bound

            # This is from the lda-c code...
            if delta < 0:
                max_iter_e_step *= 2

            # check for convergence
            if delta > 0 and delta < tolerance and (i+1) >= max_epochs:
                break

            if vocab is not None and display_topics:
                self.print_topics(vocab)

    def init_parameters(self, X, initital_smoothing, n_initial_docs=1, sequential_init=False):
        """
        Initialize parameters using recommended values from the original LDA paper
        :param X: the data (as above)
        :param initial_smoothing: the amount of smoothing to use in initializing the topics
        :param n_initial_docs: the number of documents to use to initialize each topic
        :param sequential_init: if True, take the first n_initial_docs * K documents for initialization
        :return: None
        """

        phi_total = np.zeros([self.V, self.K])
        if sequential_init:
            random_docs = list(range(n_initial_docs * self.K))
        else:
            random_docs = list(np.random.choice(np.arange(self.D), size=n_initial_docs * self.K, replace=False))
        # initialize each topic with word counts from a subset of documents
        for k in range(self.K):
            docs = random_docs[k * n_initial_docs : (k+1) * n_initial_docs]
            for d in docs:
                for w, c in X[d]:
                    phi_total[w, k] += c
        # smooth the counts
        phi_total += initital_smoothing
        # compute the corresponding topics
        self.log_betas = self.compute_log_betas_mle(phi_total)
        self.gammas = np.zeros([self.D, self.K])

    def compute_log_betas_mle(self, phi_total):
        """
        M-step for topics: compute the values of log betas to maximize the bound
        :param phi_total: np.array (V,K): Expected number of each type of token assigned to each class k
        :return: np.array (V, K): log(beta)
        """
        # sum counts over vocbaulary
        topic_word_totals = np.sum(phi_total, axis=0)
        # compute new optimal values for log betas
        log_betas = np.log(phi_total) - np.log(topic_word_totals)
        # avoid negative infinities
        log_betas[phi_total == 0] = -100
        return log_betas

    def update_parameters(self, X, max_inner_iterations=20, inner_tol=1e-6, update_global_params=True):
        """
        Do one epoch of updates for all parameters
        :param X: the data (D x V np.array)
        :param max_inner_iterations: the maximum number of iterations for optimizing document parameters
        :param inner_tol: the tolerance for optimizing  document parameters
        :param update_global_params: if True, update the weights and alpha (hyperparameter)
        :return: None
        """
        self.bound = 0
        self.sum_d__ll_per_token = 0
        phi_total = np.zeros_like(self.log_betas)

        # make one update for each document
        for d in range(self.D):
            if d % 1000 == 0 and d > 0:
                print(d)
            counts_d = X[d]

            # optimize the phi and gamma parameter for this document
            bound, phi_d, gammas = self.update_parameters_for_one_item(counts_d, max_iter_d=max_inner_iterations, tol_d=inner_tol)
            self.gammas[d, :] = gammas

            # only need to store the running sum of phi over the documents
            N_d = 0
            for n, (w, c) in enumerate(counts_d):
                phi_total[w, :] += c * phi_d[n, :]
                N_d += c

            # add the contribution of this document to the bound
            self.bound += bound
            self.sum_d__ll_per_token += bound / float(N_d)

        if update_global_params:
            # finally update the topic-word distributions and hyperparameters
            self.log_betas = self.compute_log_betas_mle(phi_total)
            self.update_alpha()


    def update_parameters_for_one_item(self, count_tuples, max_iter_d=20, tol_d=1e-6):
        """
        Update gamma and compute updates for beta and the bound for one document        
        :param counts: the word counts for the corresponding document (length-V np.array)
        :param max_iter_d: the maximum number of epochs for this inner optimization
        :param tol_d: the tolerance required for convergence of the inner optimization problem
        :return: (contribution to the bound, phi values for this doc, gammas for this doc)
        """

        # unzip counts into lists of word indices and counts of those words
        word_indices, counts = zip(*count_tuples)
        # convert the lists into vectors of the required shapes
        word_indices = np.reshape(np.array(word_indices, dtype=np.int32), (len(word_indices), ))
        counts = np.reshape(np.array(counts, dtype=np.int32), (len(word_indices), ))
        count_vector_2d = np.reshape(np.array(counts), (len(word_indices), 1))
        
        # count the total number of words
        N_d = int(count_vector_2d.sum())
        # and the number of distinct word types
        n_word_types = len(word_indices)

        # initialize gamma values to alpha + 1/K
        gammas  = self.alpha + N_d * np.ones(self.K) / float(self.K)
        # initialize phis to 1/K; only need to consider each word type
        phi_d  = np.ones([n_word_types, self.K]) / float(self.K)

        # do the optimization in a cython function
        bound = update_params_for_one_item(self.K, n_word_types, word_indices, counts, gammas, phi_d, self.log_betas, self.alpha, max_iter_d, tol_d)

        return bound, phi_d, gammas

    def update_alpha(self, newton_thresh=1e-5, max_iter=1000):
        """
        Update hyperparameters of p(\theta) using Netwon's method [ported from lda-c]
        :param newton_thresh: tolerance for Newton optimization
        :param max_iter: maximum number of iterations
        :return: None
        """

        init_alpha = 100
        log_alpha = np.log(init_alpha)

        psi_gammas = psi(self.gammas)
        psi_sum_gammas = psi(np.sum(self.gammas, axis=1))
        E_ln_thetas = psi_gammas - np.reshape(psi_sum_gammas, (self.D, 1))
        sum_E_ln_theta = np.sum(E_ln_thetas)  # called ss (sufficient statistics) in lda-c

        # repeat until convergence
        print("            alpha\tL(alpha)\tdL(alpha)")
        for i in range(max_iter):
            alpha = np.exp(log_alpha)
            if np.isnan(alpha):
                init_alpha *= 10
                print("warning : alpha is nan; new init = %0.5f" % init_alpha)
                alpha = init_alpha
                log_alpha = np.log(alpha)

            L_alpha = self.compute_L_alpha(alpha, sum_E_ln_theta)
            dL_alpha = self.compute_dL_alpha(alpha, sum_E_ln_theta)
            d2L_alpha = self.compute_d2L_alpha(alpha)
            log_alpha = log_alpha - dL_alpha / (d2L_alpha * alpha + dL_alpha)

            print("alpha maximization: %5.5f\t%5.5f\t%5.5f" % (np.exp(log_alpha), L_alpha, dL_alpha))
            if np.abs(dL_alpha) <= newton_thresh:
                break

        self.alpha = np.exp(log_alpha)

    def compute_L_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (gammaln(self.K * alpha) - self.K * gammaln(alpha)) + (alpha - 1) * sum_E_ln_theta

    def compute_dL_alpha(self, alpha, sum_E_ln_theta):
        return self.D * (self.K * psi(self.K * alpha) - self.K * psi(alpha)) + sum_E_ln_theta

    def compute_d2L_alpha(self, alpha):
        return self.D * (self.K ** 2 * polygamma(1, self.K * alpha) - self.K * polygamma(1, alpha))

    def compute_perplexity(self):
        """
        Convert the bound into perplexity
        :return: an upper bound on perplexity (float)
        """
        return np.exp(-self.sum_d__ll_per_token / float(self.D))

    def get_document_representations(self):
        """
        Return the parameter vectors of the per-document variational distributions
        """
        return self.gammas

    def get_topics(self):
        """ Get the current topic distributions"""
        return np.exp(self.log_betas)

    def evaluate(self, X, max_inner_iterations=20, inner_tol=1e-6):
        """
        Evaluate perplexity on a new batch of data
        Note that this requires optimizing document-specific parameters, and will overwrite
        the optimized values for the training data
        :param X: a new matrix of word counts with the same vocabulary (in the same order) as the training data
        :param tolerance: tolerance for convergence (relative change) (int)
        :param max_iter: minimum number of iterations (int)
        :return: an estimate of perplexity
        """
        # initialize variables for monitoring convergence
        prev_bound = -np.inf
        delta = np.inf

        # overwrite number of documents
        self.D = len(X)

        # overwrite gammas
        self.gammas = np.zeros([self.D, self.K])

        # optimize all documents once
        self.update_parameters(X, max_inner_iterations=max_inner_iterations, inner_tol=inner_tol, update_global_params=False)

        prev_bound = self.bound
        print('%0.3f\t%0.3f\t%0.5f' % (self.bound, self.sum_d__ll_per_token / float(self.D), delta))

        # return the perplexity (an approximation based on the variational bound)
        return self.compute_perplexity()

    def load_parameters(self, model_file):
        """
        Set the model parameters based on a saved model
        :param model_file: The saved model model_file
        :return: None
        """
        data = np.load(model_file)
        self.log_betas = data['log_betas']
        V, K = self.log_betas.shape
        self.K = K
        self.V = V
        self.alpha = data['alpha']

    def save_parameters(self, output_file):
        """
        Save the model to a file
        :param output_file: The file to save to
        :return: None
        """
        print("Saving final values")
        np.savez_compressed(output_file, log_betas=self.log_betas, alpha=self.alpha)

    def print_topics(self, vocab, n_words=8):
        """
        Display the top words in each topic
        :param vocab: a list of words in the vocabulary
        """
        for k in range(self.K):
            order = list(np.argsort(self.log_betas[:, k]).tolist())
            order.reverse()
            print("%d %s" % (k, ' '.join([vocab[i] for i in order[:n_words]])))


cdef double update_params_for_one_item(int K, int n_word_types, int[:] word_indices, int[:] counts, double[:] gammas, double[:, ::1] phi_d, double[:, ::1] log_betas, double alpha, int max_iter_d, double tol_d):
    """
    Optimize the per-document variational parameters for one document
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of word_indices)
    :param word_indices: a typed memory view of the indices of the words in the document
    :param counts: the corresponding counts of each word in the document
    :param gammas: the variational parameters of this document to be updated (length-K)
    :param phi_d: n_word_types x K memory view of expected distribution of topics for each word
    :param log_betas: V x K memoryview of the current value of the log of the topic distributions
    :param alpha: current value of the hyperparmeter alpha
    :param max_iter_d: the maximum number of iterations for this inner optimization loop
    :param tol_d: the tolerance required for convergence of this inner optimization loop
    """
    cdef int i = 0
    cdef int k = 0
    cdef int n, w, c

    cdef double prev_bound = -1000000.0
    cdef double bound = 0.0
    cdef double delta = tol_d
    cdef double[:] psi_gammas = np.empty(K)
    cdef double[:] log_phi_dn = np.empty(K)
    cdef double[:] new_phi_dn = np.empty(K)
    cdef double log_sum_k__phi_dn

    while k < K:
        psi_gammas[k] = digamma(gammas[k])
        k += 1

    # repeat until convergence
    while i < max_iter_d and  delta >= tol_d:
        # process all the word index: count pairs in this documents
        n = 0
        while n < n_word_types:
            w = word_indices[n]
            c = counts[n]

            # compute an unnormalized log_phi_dn
            k = 0
            while k < K:
                log_phi_dn[k] = log_betas[w, k] + psi_gammas[k]
                k += 1

            # compute normalizing constant
            log_sum_k__phi_dn = log_sum_exp_vec(log_phi_dn, K)

            # normalize to get new phi_dn
            k = 0
            while k < K:
                new_phi_dn[k] = exp(log_phi_dn[k] - log_sum_k__phi_dn)
                # update gammas with the difference between new and old (times count value)
                gammas[k] += c * (new_phi_dn[k] - phi_d[n, k])
                # store the new phi values for this word
                phi_d[n, k] = new_phi_dn[k]
                k += 1

            k = 0
            while k < K:
                psi_gammas[k] = digamma(gammas[k])
                k += 1

            n += 1

        # compute the part of the variational bound corresponding to this document
        bound = compute_bound_for_one_item(K, n_word_types, word_indices, counts, alpha, gammas, psi_gammas, phi_d, log_betas)

        # compute the relative change in the bound
        delta = (prev_bound - bound) / prev_bound

        # save the new value of the bound
        prev_bound = bound
        i += 1

    return bound


cdef double compute_bound_for_one_item(int K, int n_word_types, int[:] word_indices, int[:] count_vector, double alpha, double[:] gammas, double[:] psi_gammas, double[:, ::1] phi_d, double[:, ::1] log_betas):
    """
    Compute the parts of the variational bound corresponding to one document
    :param K: the number of topics
    :param n_word_types: the number of word types in this document (length of count_vector)
    :param word_indices: a vector of vocabulary indices of the word types in this document
    :param count_vector: the corresponding vector of counts of each word type
    :param alpha: the current value of the hyperparameter alpha
    :param gammas: the current value of gammas for this document
    :param psi_gammas: pre-computed values of psi(gammas)
    :param phi_d: the expected distribution of topics for each word type in this document
    :param log_betas: the current value of the log of the topic distributions
    """
    cdef double bound = 0.0

    # declare a new memory view to store psi(gammas)
    cdef double[:] E_ln_theta = np.empty(K)
    cdef double sum_gammas = 0
    cdef double psi_sum_gammas

    cdef int j
    cdef int k = 0
    while k < K:
        sum_gammas += gammas[k]
        k += 1
    psi_sum_gammas = digamma(sum_gammas)

    k = 0
    while k < K:
        E_ln_theta[k] = psi_gammas[k] - psi_sum_gammas
        k += 1

    bound += log_Gamma(K * alpha)
    bound -= K * log_Gamma(alpha)
    bound -= log_Gamma(sum_gammas)

    k = 0
    while k < K:
        bound += (alpha-1) * E_ln_theta[k]
        bound += log_Gamma(gammas[k])
        bound -= (gammas[k] - 1) * E_ln_theta[k]
        j = 0
        while j < n_word_types:
            bound += count_vector[j] * phi_d[j, k] * (E_ln_theta[k] + log_betas[word_indices[j], k] - log(phi_d[j, k]))
            j += 1
        k += 1
    return bound



