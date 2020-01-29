#!/usr/bin/env python3

import argparse
import os
import textract
import re
import spacy
import codecs


# # flow control function
def extract_sentences(src, model):

    print('Model: ' + str(model) + '...')

    # # load NLP library object
    nlp = spacy.load(model)

    source_file = os.path.basename(src)
    base = os.path.splitext(source_file)[0]

    with codecs.open(src,
                    "r",
                    encoding='utf-8',
                    errors='ignore') as fdata:

        data=fdata.read().replace('\n', '')
        clean_data = data.replace("\\n", " ").replace("\\t"," ")

        document = nlp(clean_data)

        for sent in document.sents:
            print(str(base), ", ", str(sent).strip())



if __name__ == "__main__":

    # # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--src", help="Choose the text file to process.")
    parser.add_argument("--model", nargs='?', default='en_core_web_sm',\
    help="Choose NER model.")

    args = parser.parse_args()

    # # run puppy, run
    extract_sentences(args.src, args.model)
