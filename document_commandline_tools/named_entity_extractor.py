#!/usr/bin/env python

import argparse # # inorder to port to command line
import os # # system controls
import textract # # used to parse text files
import re # # string operations, mostly data cleaning
import spacy # # industrial strength NLP engine
from spacy.symbols import nsubj, VERB

# # function to handle input larger than 1000000 character
# # takes a string (not NLP object) and a list size as an input
def textSplit(text, size):
    text_splits = []

    for i in range(0, len(text), size):
        text_splits.append(text[i:i + size])

    return text_splits


# # get list of single words
# # takes an NLP object as input
# # if needed: https://spacy.io/api/annotation#pos-tagging
def getTokens(document):
    tokenList = []

    for token in document:
        tup = (token.text,[token.lemma_, token.pos_, token.tag_, token.dep_,\
              token.shape_, token.is_alpha, token.is_stop])
        tokenList.append(tup)

    return tokenList


# # get list of noun-chunks from an NLP object
# # takes an NLP object as input
def getChunks(document):
    chunkList = []

    for chunk in document.noun_chunks:
        tup = (chunk.text, [chunk.root.text, chunk.root.dep_, \
          chunk.root.head.text])
        chunkList.append(tup)

    return chunkList


# # get list of items that preceed the current token
# # takes an NLP object as input
def getChildren(document):
    childList = []

    for token in document:

        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children], '\n')


# # iterates through document for matches of certain types
# # entity_types defined in extractNERs() function
# # takes an NLP object as input and a list of entity types
def getNamedEntities(document):
    print('Searching for named_entity...')

    valid_ent = re.compile("^[a-zA-Z0-9_]*$")
    ENTITY_LIST = []

    for entity in document.ents:
        NER_tup = (str(entity.label_).upper(), re.sub('[^A-Za-z0-9]+', ' ', str(entity.text)))
        ENTITY_LIST.append(NER_tup)

    print('named_entity search COMPLETE.')
    print('Found ' + str(len(ENTITY_LIST)) + '\n')
    return ENTITY_LIST


# # flow control function
# # takes a a commandline argument (file)
def extractNERs(input_file):

    # # model = 'en_core_web_lg'
    model = 'en_core_web_md'
    print('Loading language library: ' + str(model) + '...')

    # # load NLP library object
    nlp = spacy.load(model)
    print('Library load COMPLETE.\n')

    # # file operations to manage printing
    # # probably need to get full path as well...
    source_file = os.path.basename(input_file)
    base = os.path.splitext(source_file)[0]

    # # need to make simple file handling more robust
    # # it may be the case that this is better suited for another script,
    # # or should be called from main. TDB.

    print('Reading file: ' + str(source_file) + '...')

    # # create nlp object from input_file
    text = str(textract.process(input_file,
            method='tesseract',
            language='en'))

    print('Read COMPLETE.\n')


    # # breaks files into manageable chunks
    # # may be suited for a separate function call,
    # # though likely called only once
    if len(text) > 999999:
        text_list = []

        # # break document into fewest possible splits:
        splits = math.ceil(len(text)/999999)

        print('Document with length ' + str(len(text)) + \
            ' must be distributed between ' + str(splits) + ' splits...')

        # # add all splits to new list
        [text_list.append(x) for x in textSplit(text, math.ceil(len(text)/splits))]
        print('Document split COMPLETE.\n')
    else:
        splits = 1
        # # if document under 10000000 chars, just store
        text_list = []
        text_list.append(text)

    # # index the chunks of the file
    processing_count = 1

    # # process all string chunks
    for string_chunk in text_list:

        print('Converting file to spacy object: ' + str(source_file) + '...')
        print('Part ' + str(processing_count) + ' of ' + str(splits))
        document = nlp(string_chunk)
        print('Convert COMPLETE.\n')

        # # increment the document chunk
        processing_count += 1

        # # Some print testing
        # [print(x) for x in getTokens(document)]
        # [print(x) for x in getChunks(document)]
        # getChildren(document)

        getNamedEntities(document)
        [print(x) for x in getNamedEntities(document)]

if __name__ == "__main__":

    # # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--input", help="Choose the text file to process.")
    args = parser.parse_args()

    # # run puppy, run
    extractNERs(args.input)
