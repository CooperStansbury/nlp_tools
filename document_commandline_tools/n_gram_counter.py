#!/usr/bin/python
import sys
import re
import itertools
from collections import Counter
import textract # used to parse PDF files
import argparse
import os
import operator


# main function call
def wordCounter(input_file, n):

    # convert CLI arg to int
    n = int(n)

    # basic file operations
    source_file = os.path.basename(input_file)
    base = os.path.splitext(source_file)[0]
    print('*** PRINTING RESULTS FOR: ' + str(base) + ' ***')

    # simple word parsing and cleaning stored in all_words list
    all_words = []
    with open(input_file,'r') as f:
        for line in f:
            for word in line.split():
                # # these tests would be better moved elsewhere
                word = re.sub(r'[^\w\s]',' ',word)
                word = word.strip()
                word = word.lower()
                all_words.append(word)

    # get combinations based on gram size CLI arg
    gram_list = []

    # iterate through words and save n grams up to user input n
    for j in range(1,n+1):
        [gram_list.append(" ".join(x)) for x in zip(*[all_words[i:] for i in range(j)])]

    # save frequency counts
    counts = Counter(gram_list)

    # convert to tuples and sort
    sorted_counter = sorted(counts.items(), key=operator.itemgetter(1))

    # print to stdout to use with bash
    [print(str(x[0]) + ', ' + str(x[1])) for x in sorted_counter]


if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--gram_size", help="grams up to size x")
    parser.add_argument("--input_file", help="choose the text file to process.")
    args = parser.parse_args()

    # run puppy, run
    wordCounter(args.input_file, args.gram_size)
