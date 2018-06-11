# /bin/python2.7
import os
import sys
import numpy as np
sys.path.append('../../matchzoo/inputs')
sys.path.append('../../matchzoo/utils')
from preparation import *
from preprocess import *


if __name__ == '__main__':

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '../../data/toy_example/classification/sample.txt'

    basedir = os.path.dirname(path)

    # transform query/document pairs into corpus file and relation file
    prepare = Preparation()
    corpus, rels = prepare.run_with_one_corpus(path)
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(os.path.join(basedir, 'corpus.txt'), corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.4, 0.3, 0.3])
    prepare.save_relation(os.path.join(basedir, 'relation_train.txt'), rel_train)
    prepare.save_relation(os.path.join(basedir, 'relation_valid.txt'), rel_valid)
    prepare.save_relation(os.path.join(basedir, 'relation_test.txt'), rel_test)
    print('preparation finished ...')

    # Prerpocess corpus file
    preprocessor = Preprocess()

    dids, docs = preprocessor.run(os.path.join(basedir, 'corpus.txt'))
    preprocessor.save_word_dict(os.path.join(basedir, 'word_dict.txt'))
    preprocessor.save_words_stats(os.path.join(basedir, 'word_stats.txt'))

    fout = open(os.path.join(basedir, 'corpus_preprocessed.txt'),'w')
    for inum,did in enumerate(dids):
        fout.write('%s %d %s\n'%(did, len(docs[inum]), ' '.join(map(str,docs[inum]))))
    fout.close()
    print('preprocess finished ...')

