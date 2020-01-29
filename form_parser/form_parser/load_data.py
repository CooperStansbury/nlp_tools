"""
description: functions to crawl directories and get text data as strings from
.txt files
"""

# non-local imports
import os
import re
import pandas as pd
import numpy as np

# local imports
import clean_data as cd


def get_export_date():
    """ return str of the execution date """
    import datetime
    return str(datetime.date.today())


def get_file_name(filepath):
    """ return str of filename given a path """
    return str(os.path.splitext(filepath)[0])


def get_data_from_file(filepath):
    """ retrun dict of info from filepath """
    if filepath.endswith('.txt'):
        print('Processing ', filepath)
        with open(filepath, 'r', encoding="utf8", errors='ignore') as myfile:
            data = myfile.read().replace('\n', ' ')
            return {'name': get_file_name(filepath),
                    'path':filepath,
                    'raw_text':data,
                    'export_date': get_export_date()}


def add_sequential_ids(df):
    """ return new pd.DataFrame with sequential record ID in new column """
    print('Adding sequential record ID in new column.')
    df['id'] =  np.arange(len(df))
    df['id'] += 1 # start index at 1, not 0
    return df


def get_data_from_dir(directory):
    """ returns a dataframe with filenames, paths, and unprocessed text """
    print('Getting data from: ', str(directory))
    new_rows = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            new_rows.append(get_data_from_file(filepath))
    df = pd.DataFrame(new_rows)
    print('Returning DataFrame with the following columns: ', str(df.columns))
    return df
