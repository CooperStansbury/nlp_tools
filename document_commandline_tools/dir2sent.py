#!/usr/bin/env python3

import argparse
import os
import textract
import re
import spacy
import codecs
import pandas as pd


# # flow control function
def getSentences(dir, model, dest):

    print('Model: ' + str(model) + '...')

    # # load NLP library object
    nlp = spacy.load(model)


    # iterate through directory
    for subdir, dirs, files in os.walk(dir):
        for file in files:

            filepath = subdir + os.sep + file

            if filepath.endswith('.txt'):

                rows = []

                with codecs.open(filepath,
                                "r",
                                encoding='utf-8',
                                errors='ignore') as fdata:

                    data=fdata.read().replace('\n', '')
                    clean_data = data.replace("\\n", " ").replace("\\t"," ")

                    document = nlp(clean_data)

                    for sent in document.sents:

                        new_row = {}

                        new_row['orin_file_name'] = str(filepath)
                        new_row['sentenceText'] = str(sent).strip()
                        new_row['startChar'] = str(sent.start_char).strip()

                        rows.append(new_row)

    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)


if __name__ == "__main__":

    # # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--dir", help="Choose the dir to process.")
    parser.add_argument("--model", nargs='?', default='en_core_web_sm',\
    help="Choose NER model.")
    parser.add_argument("--dest", help="Choose the dir to process.")

    args = parser.parse_args()

    # # run puppy, run
    getSentences(args.dir, args.model, args.dest)
