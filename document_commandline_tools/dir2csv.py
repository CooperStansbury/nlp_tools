#!/usr/bin/env python3

import argparse
import os
import re
import pandas as pd
import numpy as np
import nltk
import itertools
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster as sc
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import ward, dendrogram

def toDataFrame(fileDictionary):
    """ return pandas dataframe
    Note: expects input in nested dict format (output from getData)"""
    new_rows = []

    for key, value in fileDictionary.items():
        new_rows.append(value)

    df = pd.DataFrame(new_rows)
    print('\nstructure:\n ', df.dtypes)
    print('\nsummary:\n ', df.describe())

    return df


## TODO: this function is overloaded
def dir2dict(dir):
    """ returns a dict with cleaned filenames, full paths,
    and text data in multiple formats from a dictionary
    of .txt files"""

    fileDictionary = {}
    fileID = 0

    # iterate through directory
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            fileID += 1
            filepath = subdir + os.sep + file

            if filepath.endswith('.txt'):
                wordList = [] # this will include duplicates and stop words
                cleanedFileName = ''.join(e for e in file if e.isalnum())[:-3]
                fileSize = os.path.getsize(filepath) # not sure if needed

                # perform string operations on each file
                with open(filepath, 'r') as myfile:
                    data = myfile.read().replace('\n', ' ')

                    # strip special characters and make lowercase
                    for word in str(data).split(" "):
                        word = re.sub("[^a-zA-Z]+", " ", word).strip().lower()

                        # add only non-blank words
                        if not word == "":
                            wordList.append(word)

                print('file: ', cleanedFileName, 'countWords: ', len(wordList))

                # build nested dict
                fileDictionary[fileID] = {}
                fileDictionary[fileID]['name'] = cleanedFileName
                fileDictionary[fileID]['path'] = filepath
                fileDictionary[fileID]['fileSize'] = fileSize
                fileDictionary[fileID]['text'] = data
                fileDictionary[fileID]['lengthRawData'] = len(data)
                fileDictionary[fileID]['wordList'] = wordList
                fileDictionary[fileID]['countWords'] = len(wordList)
                fileDictionary[fileID]['wordList_Unique'] = list(set(wordList))
                fileDictionary[fileID]['countUniqueWords'] = len(set(wordList))
                fileDictionary[fileID]['wordListasString'] = " ".join(wordList)

    return fileDictionary


if __name__ == "__main__":

    # # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--dir", help="Choose the directory to process.")
    parser.add_argument("--dest", help="csv output filename")
    args = parser.parse_args()

    #
    dict = dir2dict(args.dir)
    df = toDataFrame(dict)
    df.to_csv(str(args.dest), index=False)
