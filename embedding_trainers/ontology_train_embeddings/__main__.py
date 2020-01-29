#!/usr/bin/env python3

import owlready2 as ow
import pandas as pd
import argparse
import os
from tabulate import tabulate
from gensim.models import Word2Vec
import spacy
from sklearn.decomposition import PCA
from matplotlib import pyplot

# load space model
nlp = spacy.load("en_core_web_sm")

def get_local_ontology_from_file(ontology_file):
    """ return ontology class from a local OWL file """
    return ow.get_ontology("file://" + ontology_file).load()


def get_class_from_ontolgy(ontology, string):
    """ return ontology class from string """
    return ontology.search_one(iri=string)


def get_ancestors_normalized(onto, ancestor_list):
    """ convert ancestor list to a list of strings """
    label_list = ["".join(set(ancestor.label)) for ancestor in list(ancestor_list)]
    label_list = list(filter(None, label_list)) # remove blanks
    return label_list


def get_features(onto):
    """ get language features from ontology object, return dataframe """

    new_rows = []

    for cl in onto.classes():
        parent = cl.is_a
        ancestor_list = get_ancestors_normalized(onto, cl.ancestors())

        if len(parent) > 1:
            row = {
                "class":"".join(set(cl.label)),
                "parent_class":"".join(set(parent[0].label)),
                "definition":"".join(set(cl.IAO_0000115)),
                "ancestor_list":ancestor_list,
            }
            new_rows.append(row)

    df = pd.DataFrame(new_rows)
    return(df)


def add_doc_objects(df):
    """ train embeddings on definitions """

    def apply_get_doc_objects_from_def(row):
        """ return spacy doc object from a text field """
        doc = nlp(str(row['definition']).lower())
        return doc

    df['doc'] = df.apply(lambda row: apply_get_doc_objects_from_def(row), axis=1)
    return df


def get_token_list(df):
    """ return a new column with list of tokens """

    def apply_split_list(row):
        """ split each row into a list of tokens """
        token_list = []
        for token in row['doc']:
            token_list.append(token.text)
        return token_list

    df['token_list'] = df.apply(lambda row: apply_split_list(row), axis=1)
    return df


def train_embeddings(df):
    """ train emneddings (Word2Vec) on definitions """

    sentences = df['token_list']

    model = Word2Vec(sentences,
                 min_count=3,
                 size=100,      # output dimensionality
                 workers=2,     # parallelization
                 window=10,     # context window
                 iter=30)       # epochs

    return model


def plot_results(model):
    """ plot PCA of the results """

    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
    	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description='Extract terms from local .OWL owl_file')
    parser.add_argument("--owl_file", help="owl file to query")
    parser.add_argument("--print", action='store_true', \
            help="if present: print output to console")
    args = parser.parse_args()


    source_owl_file = os.path.abspath(args.owl_file)
    onto = get_local_ontology_from_file(source_owl_file)
    df = get_features(onto)
    df = add_doc_objects(df)
    df = get_token_list(df)

    print(df.head())

    model = train_embeddings(df)
    plot_results(model)
