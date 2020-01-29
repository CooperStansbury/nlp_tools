#!/usr/bin/env python3

import sys
import os
import pandas as pd
import argparse

def getCounts(src, file_type, keys, dest):

    new_rows = []

    # open list of user provided keys
    key_frame = pd.read_csv(keys, delimiter=',')
    clues = key_frame['CLUES']

    # iterate through dir to find all files of specified file_type
    for subdir, dirs, files in os.walk(src):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(file_type):
                try:
                    with open(filepath, encoding='utf-8') as f:
                        file_contents = f.read()

                        # if in research dir, then 1, else 0
                        research_consent = 0
                        if str(filepath).__contains__('research_forms'):
                            research_consent = 1

                        # build row for addition into dataframe
                        row = {
                            'filename': file,
                            'word_count':len(file_contents.split(" ")),
                            'period_count':file_contents.count('.'),
                            'research_consent':research_consent,
                        }

                        # add new field for each clue
                        for clue in clues:
                            row.update({str(clue).replace(" ","_")+ \
                                    '_mentions':file_contents.count(clue)})

                        # append dicts to list
                        new_rows.append(row)
                except Exception as e:
                    print('\n ERROR PROCESSING: ', str(filepath), '\n', e, '\n')
                    continue

    df = pd.DataFrame(new_rows)

    # save file based on positional arg
    df.to_csv(dest)


if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description='file')
    parser.add_argument("--src", help="Directory of files")
    parser.add_argument("--file_type", help="File extension. Ex: '.txt'")
    parser.add_argument("--keys", help="File containing search phrases")
    parser.add_argument("--dest", help="Name of destination .csv output")
    args = parser.parse_args()

    # run puppy, run
    getCounts(args.src, args.file_type, args.keys, args.dest)
