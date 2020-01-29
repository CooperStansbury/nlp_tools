#!/usr/bin/env python3

import pandas as pd
import argparse
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
import csv

def tfIDF(src,file_type,dest):

    en_stops = set(stopwords.words('english'))
    corpus = []

    # iterate through dir to find all files of specified file_type
    for subdir, dirs, files in os.walk(src):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            file_word_list = []

            if filepath.endswith(file_type):
                try:
                    f = open(filepath, "r").read()
                    for word in f.split():
                        if word not in en_stops:
                            word = re.sub("[^a-zA-Z]+", "", word)
                            file_word_list.append(word.lower())
                    file_contents = " ".join(file_word_list)
                    corpus.append(file_contents)
                except Exception as e:
                    print('\n ERROR PROCESSING: ', str(filepath), '\n', e, '\n')
                    continue

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    feature_list = vectorizer.get_feature_names()

    with open(dest, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(feature_list)


if __name__ == "__main__":
    # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--src", help="Directory of files")
    parser.add_argument("--file_type", help="File extension. Ex: '.txt'")
    parser.add_argument("--dest", help=".csv file to train a model")
    args = parser.parse_args()

    # run puppy, run
    tfIDF(args.src, args.file_type, args.dest)
