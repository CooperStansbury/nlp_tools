#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from gensim.models import Word2Vec
import spacy
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re
import dask.dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
from datetime import datetime


# load space model
nlp = spacy.load("en_core_web_md")


def get_date():
    """ return  today's date as an appendable string"""
    return datetime.today().strftime('%Y-%m-%d')


def load_dir(dir_path):
    """ load text files into a dataframe """

    new_rows = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            filepath = subdir + os.sep + file
            filename = file.split(".")[0]
            file_id = filename.split("_")[1]
            with open(filepath, 'r') as myfile:
                data = myfile.read()
                row = {
                    "name" : filename,
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
        clean_text = clean_text.replace("_", "") # strip signature lines
        return clean_text

    df['clean_data'] = df.apply(lambda row: apply_clean_corpus(row), axis=1)
    return df


def get_doc_objects(df):
    """ create spacy doc objects """

    def parallel_apply_get_doc(row):
        """ return spacy doc object from a text field """
        doc = nlp(str(row['clean_data']).lower())
        return doc

    df['doc'] = dd.from_pandas(df , npartitions=cpu_count()).\
       map_partitions(
          lambda df : df.apply(
             lambda x :parallel_apply_get_doc(x),axis=1)).\
       compute(scheduler='threads')

    return df


def get_sentence_lists(df):
    """ return list of sentences """

    def apply_parse_sents(row):
        """ return list of sentences from doc object field;
        each item will be token span """

        return list(row['doc'].sents)

    df['sent_list'] = df.apply(lambda row:apply_parse_sents(row),axis=1)
    return df


def normalize_corpus(df):
    """ return a list of tokenized sentences """
    corpus = []

    for sent_list in df['sent_list']:
        for sent in sent_list:
            to_add = []
            for token in sent:
                to_add.append(token.text)
            corpus.append(to_add)

    # [print(x, '\n') for x in corpus]
    return corpus



def train_embeddings(list):
    """ train emneddings (Word2Vec) on definitions """

    model = Word2Vec(list,
                 min_count=3,
                 size=100,      # output dimensionality
                 workers=2,     # parallelization
                 window=10,     # context window
                 iter=30)       # epochs

    return model


def train_glove_embeddings(list, learning_rate=0.05):
    """ train GloVe embeddings """

    # TODO: build custom implementation?

    # from glove import Corpus, Glove
    # corpus = Corpus()
    # corpus.fit(list, window=10)
    # glove = Glove(no_components=100, learning_rate=learning_rate)
    # glove = Glove(no_components=100, learning_rate=learning_rate)
    # glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    # glove.add_dictionary(corpus.dictionary)


def print_most_similar(model, word_string):
    """ print most similar (closest) words """
    print("--------------------------------", word_string, "--------------------------------")
    print("Words most similar to", word_string)
    [print(x) for x in model.wv.most_similar(word_string, topn=20)]


def save_model(model, location_path):
    """ save a model, expects .bin """
    model.wv.save_word2vec_format(location_path)


def plot_results(model, label=False):
    """ plot PCA of the results """

    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    if label:
        for i, word in enumerate(words):
        	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
    else:
        pyplot.show()


def informed_consent_domain_test(model):
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
        print_most_similar(model, t)

if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description='Extract terms from local .OWL owl_file')
    parser.add_argument("--text_dir", help="owl file to query")
    parser.add_argument("--save", action='store_true', \
            help="if present: require --output_path flag ")
    parser.add_argument("--output_path",  nargs='?', help="path to save output")
    args = parser.parse_args()

    df = load_dir(args.text_dir)
    df = clean_text(df)
    # df = df.sample(10)
    df = get_doc_objects(df)
    df = get_sentence_lists(df)
    corpus = normalize_corpus(df)

    # model = train_glove_embeddings(corpus)

    model = train_embeddings(corpus)

    informed_consent_domain_test(model)
    plot_results(model)

    if args.save and args.output_path:
        save_model(model, args.output_path)
    elif args.save:
        print("missing --output_path flag")

    # print(df.head())
