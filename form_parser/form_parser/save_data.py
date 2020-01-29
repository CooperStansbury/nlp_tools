"""
description: functions to save files and metadata once data cleaning is done
"""

# non-local imports
import re
import pandas as pd
import os.path
import csv

# local imports
import load_data as ld


# def normalize_sentences(df, clean_sentence_column):
#     """ return a new dataframe with each sentence on it's row """
#     new_rows = []
#     for index, row in df.iterrows():
#         for sentence in row[clean_sentence_column]:
#             row = {
#                 'id' : row['id'],
#                 'file_name' : row['name'],
#                 'sentence' : sentence,
#                 'export_date' : row['export_date']}
#             new_rows.append(row)
#     return pd.DataFrame(new_rows)


def format_save_path(filename, ext):
    """ return string with path for new file """
    output_dir = '../output/'
    date = str(ld.get_export_date())
    return output_dir + filename + date + ext


def check_save_path(save_path):
    """ return TRUE if file name exists """
    return os.path.isfile(save_path)


def get_sentence_groups(sent_list, group_size):
    """ return chunks of size n from list of sentences """
    import itertools
    iters = itertools.tee(sent_list, group_size)
    for i, it in enumerate(iters):
        next(itertools.islice(it, i, i), None)
    return zip(*iters)


def save_sentences_to_file(filename, df, clean_sentence_column, group_size=2):
    """ save sentences to csv with filename """
    save_path = format_save_path(filename, '.csv')
    if check_save_path(save_path):
        print(save_path, ' already exists!')
    else:
        with open(save_path, 'w') as outcsv:
            writer = csv.writer(outcsv,
                                delimiter=',',
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')

            writer.writerow(['file_id','sentence'])

            for index, row in df.iterrows():
                sent_list = row[clean_sentence_column]
                grouped_sents = get_sentence_groups(sent_list, group_size)

                for group in grouped_sents:
                    phrase = ' '.join(group)

                ## this is a first appraoch limited to sent pairs
                # for first, second in zip(sent_list, sent_list[1:]):
                #     phrase = str(first) + str(second)
                #     print(phrase, '\n')


                    file_id = "[[fileID:" + str(row['id']) + "]]"
                    writer.writerow([file_id, phrase])
    print('saved file', str(save_path))


def drop_columns(df, drop_columns):
    """ return a new dataframe without spacy objects """
    return df.drop(drop_columns, axis=1)


def save_sentence_metadata(filename, df):
    """ save metadata to file """
    save_path = format_save_path(filename, '.csv')
    if check_save_path(save_path):
        print(save_path, ' already exists!')
    else:
        df.to_csv(save_path, index=False)
        print('saved file', str(save_path))


# # TODO: need to reduce spacy_doc objects
# def pickle_dataframe(filename, df):
#     """ save dataframe so that it can be accessed again """
#     save_path = format_save_path(filename, '.pkl')
#     if check_save_path(save_path):
#         print(save_path, ' already exists!')
#     else:
#         df.to_pickle(save_path)
