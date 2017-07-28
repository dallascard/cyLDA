import os
import re
import sys
from optparse import OptionParser
from collections import Counter

import numpy as np
from scipy import sparse
from spacy.en import English

import file_handling as fh


def main():
    usage = "%prog train.json test.json output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--vocab_size', dest='vocab_size', default=2000,
                      help='Size of the vocabulary (by most common): default=%default')
    parser.add_option('--malletstop', action="store_true", dest="malletstop", default=False,
                     help='Use Mallet stopwords: default=%default')
    parser.add_option('--replace_num', action="store_true", dest="replace_num", default=False,
                      help='Replace numbers with <NUM>: default=%default')
    parser.add_option('--drop_num', action="store_true", dest="drop_num", default=False,
                      help='Exclude all but letters: default=%default')
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas: default=%default')

    (options, args) = parser.parse_args()

    if len(args) != len(usage.split())-1:
        print("Please provide all input arguments")

    train_infile = args[0]
    test_infile = args[1]
    output_dir = args[2]
    vocab_size = int(options.vocab_size)
    use_mallet_stopwords = options.malletstop
    replace_num = options.replace_num
    drop_num = options.drop_num
    lemmatize = options.lemmatize

    if not os.path.exists(output_dir):
        sys.exit("Error: output directory does not exist")

    preprocess_data(train_infile, test_infile, output_dir, vocab_size, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, drop_num=drop_num, lemmatize=lemmatize)


def preprocess_data(train_infile, test_infile, output_dir, vocab_size, use_mallet_stopwords=False, replace_num=False, drop_num=False, lemmatize=False):

    print("Loading SpaCy")
    parser = English()
    train_X, train_vocab, train_items, train_dat = load_and_process_data(train_infile, vocab_size, parser, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, drop_num=drop_num, lemmatize=lemmatize)
    test_X, _, test_items, test_dat = load_and_process_data(test_infile, vocab_size, parser, vocab=train_vocab, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, drop_num=drop_num, lemmatize=lemmatize)
    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_items, os.path.join(output_dir, 'train.items.json'), sort_keys=False)
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'), sort_keys=False)
    fh.write_list_to_text(train_vocab, os.path.join(output_dir, 'train.vocab.txt'))
    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_items, os.path.join(output_dir, 'test.items.json'), sort_keys=False)
    # export files that could be used for a comparison with the original lda-c code
    fh.write_list_to_text(train_dat, os.path.join(output_dir, 'train.dat'))
    fh.write_list_to_text(test_dat, os.path.join(output_dir, 'test.dat'))
    fh.write_list_to_text(train_vocab, os.path.join(output_dir, 'train.vocab.txt'))


def load_and_process_data(infile, vocab_size, parser, strip_html=False, vocab=None, use_mallet_stopwords=False, replace_num=False, drop_num=False, lemmatize=False):

    mallet_stopwords = None
    if use_mallet_stopwords:
        print("Using MALLET stopwords")
        mallet_stopwords = fh.read_text('mallet_stopwords.txt')
        mallet_stopwords = {s.strip() for s in mallet_stopwords}

    print("Reading data files")
    item_dict = fh.read_json(infile)
    n_items = len(item_dict)

    parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    keys = list(item_dict.keys())
    keys.sort()
    for i, k in enumerate(keys):
        item = item_dict[k]
        if i % 1000 == 0 and i > 0:
            print(i)

        text = item['text']

        if strip_html:
            # remove each pair of angle brackets and everything within them
            text = re.sub('<[^>]+>', '', text)

        parse = parser(text)
        # remove white space from tokens
        if lemmatize:
            words = [re.sub('\s', '', token.lemma_) for token in parse]
        else:
            words = [re.sub('\s', '', token.orth_) for token in parse]
        # convert to lower case and drop empty strings
        words = [word.lower() for word in words if len(word) > 0]
        # remove stop words
        if use_mallet_stopwords:
            words = [word for word in words if word not in mallet_stopwords]
        # remove tokens that don't contain letters or numbers
        if drop_num:
            words = [word for word in words if re.match('^[a-zA-A]*$', word) is not None]
        else:
            words = [word for word in words if re.match('[a-zA-A0-9]', word) is not None]
        # convert numbers to a number symbol
        if replace_num:
            words = ['<NUM>' if re.match('[0-9]', word) is not None else word for word in words]
        # store the parsed documents
        parsed.append(words)
        # keep track fo the number of documents with each word
        word_counts.update(words)
        doc_counts.update(set(words))

    print("Size of full vocabulary=%d" % len(word_counts))

    if vocab is None:
        most_common = doc_counts.most_common(n=vocab_size)
        words, counts = zip(*most_common)
        print("Most common words:")
        for w in range(20):
            print(words[w], doc_counts[words[w]], word_counts[words[w]])
        vocab = list(words)
        vocab.sort()

    vocab_index = dict(zip(vocab, range(vocab_size)))

    X = np.zeros([n_items, vocab_size], dtype=int)

    dat_strings = []

    print("First document:")
    print(' '.join(parsed[0]))

    counter = Counter()
    print("Converting to count representations")
    count = 0
    for i, words in enumerate(parsed):
        indices = [vocab_index[word] for word in words if word in vocab_index]
        counter.clear()
        counter.update(indices)
        # only include non-empty documents
        if len(counter.keys()) > 0:
            # udpate the counts
            values = list(counter.values())
            X[np.ones(len(counter.keys()), dtype=int) * count, list(counter.keys())] += values
            # also export to .dat format for Blei's lda-c code
            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(int(k)) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)
            count += 1

    print("Found %d non-empty documents" % count)

    # drop the items that don't have any words in the vocabualry
    X = np.array(X[:count, :], dtype=int)

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)

    return sparse_X, vocab, keys, dat_strings


if __name__ == '__main__':
    main()
