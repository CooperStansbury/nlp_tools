from __future__ import print_function
import argparse
import gensim
import re
import os
import pandas as pd
import random

from glove import Glove
from glove import Corpus

def load_dir(dir_path):
    """ load text files into a dataframe """

    new_rows = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            filepath = subdir + os.sep + file
            with open(filepath, 'r', encoding = "ISO-8859-1") as myfile:
                data = myfile.read()
                row = {
                    "name" : file,
                    "path" : filepath,
                    "data" : data
                       }
                new_rows.append(row)
    return pd.DataFrame(new_rows)


def clean_text(df):
    """ function to manage cleaning operations on text fields """

    def apply_clean_corpus(row):
        """ return new column with cleaned text """
        encoded_text = row['data'].encode(encoding = 'ascii',errors = 'replace')
        decoded_text = encoded_text.decode(encoding='ascii',errors='strict')
        remove_funky_chars = str(decoded_text).replace("?", " ")
        lower_case = str(remove_funky_chars).lower().strip()
        clean_text = re.sub('[^A-Za-z0-9]+', ' ', lower_case).strip()
        clean_text = re.sub(' +', ' ', clean_text) # strip redundant whitespace
        return clean_text

    df['clean_data'] = df.apply(lambda row: apply_clean_corpus(row), axis=1)
    return df


def tokenize_corpus(df):
    """ return a list of tokenized words for each file """

    def apply_naive_tokenize(row):
        """ tokenize based on whitespace """
        return row['clean_data'].split()

    df['tokenized_docs'] = df.apply(lambda row: apply_naive_tokenize(row), axis=1)
    return df


def get_corpus_model(token_list, window_size, save=True):
    """ wrapper function to build corpus using glove lib """
    print('Training the corpus model...')
    corpus_model = Corpus()
    corpus_model.fit(token_list, window=window_size)

    if save:
        corpus_model.save('corpus.model')
        print("Saved Corpus to 'corpus.model'")
    return corpus_model


def get_trained_glove(corpus_model, dim, n_epochs, threads, save=True):
    """ wrapper funtion to trian glove vectors """
    print('Training the GloVe model...')
    glove = Glove(no_components=dim, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=n_epochs,
              no_threads=threads, verbose=True)

    glove.add_dictionary(corpus_model.dictionary)

    if save:
        glove.save('glove.model')
        print("Saved Corpus to 'glove.model'")

    return glove


def informed_consent_domain_test(model, n=10):
    """ domain specific print tests for informed consent forms """
    terms = ["consent",
            "permission",
            "sample",
            "blood",
            "information",
            "risk",
            "store",
            "tissue"]

    for t in terms:
        print("------------------ Words most similar to: '{}' ".format(t))
        sims = model.most_similar(t, number=n)
        [print(x) for x in sims]


def general_test(model, corpus, n=10, n_sim=10):
    """ domain specific print tests for informed consent forms """
    print("Printing n={} random terms and their m={} closest terms in the"
        " embedding space".format(n, n_sim))

    for t in random.sample(list(corpus.dictionary), 10):
        print("------------------ Words most similar to: '{}' ".format(t))
        sims = model.most_similar(t, number=n_sim)
        [print(x) for x in sims]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a GloVe model.')

    parser.add_argument('--dir', '-d', help=('The dir of txt files to pre-process. '
                              'The pre-processed corpus will be saved '
                              'and will be ready for training.'))

    parser.add_argument('--window_size', '-w', default=10,
                        help=('Context window for training corpus object.'))

    parser.add_argument('--epochs', '-e', default=10,
                        help=('Number of epochs to train.'))

    parser.add_argument('--dimensions', '-dim', default=100,
                        help=('Dimensionality of output projection.'))

    parser.add_argument('--parallelism', '-p', action='store',
                        default=2,
                        help=('Number of parallel threads to use for training'))

    parser.add_argument('--consent_print_test', '-icf', action='store_true',
                            help=('Print tests specific to informed consent.'))

    args = parser.parse_args()

    print('Pre-processing corpus...')
    df = tokenize_corpus(clean_text(load_dir(args.dir)))

    corpus_model = get_corpus_model(df['tokenized_docs'].tolist(), args.window_size)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    glove = get_trained_glove(corpus_model=corpus_model,
                              dim=int(args.dimensions),
                              n_epochs=int(args.epochs),
                              threads=int(args.parallelism),
                              save=True)

    if args.consent_print_test:
        informed_consent_domain_test(glove)
    else:
        general_test(glove, corpus_model)
