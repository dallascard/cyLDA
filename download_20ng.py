import os
from optparse import OptionParser

from sklearn.datasets import fetch_20newsgroups

import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-d', dest='dir', default='data',
                      help='Directory in which to save data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    data_dir = options.dir

    names = ['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism']

    prefixes = set([name.split('.')[0] for name in names])

    groups = {'20ng_all': names}
    for prefix in prefixes:
        elements = [name for name in names if name.split('.')[0] == prefix]
        groups['20ng_' + prefix] = elements

    for group in groups.keys():
        if len(groups[group]) > 1:
            for subset in ['train', 'test', 'all']:
                download_articles(data_dir, group, groups[group], subset)


def download_articles(data_dir, name, categories, subset):

    data = {}
    print("Downloading articles")
    newsgroups_data = fetch_20newsgroups(subset=subset, categories=categories, remove=())

    for i in range(len(newsgroups_data['data'])):
        line = newsgroups_data['data'][i]
        data[str(len(data))] = {'text': line, 'label': newsgroups_data['target_names'][newsgroups_data['target'][i]]}

    print(len(data))
    raw_data_dir = os.path.join(data_dir, '20ng', name)
    print("Saving to", raw_data_dir)
    fh.makedirs(raw_data_dir)
    fh.write_to_json(data, os.path.join(raw_data_dir, subset + '.json'))


if __name__ == '__main__':
    main()
