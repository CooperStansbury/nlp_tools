"""
description: load, clean, and save sentences from a directory of .txt files
author: Cooper Stansbury
"""

# non-local imports
import os
import re
import pandas as pd
import numpy as np

# local imports
import load_data as ld
import clean_data as cd
import save_data as sd


if __name__ == "__main__":
    data_dir = '../data/'
    df = ld.get_data_from_dir(data_dir)
    df = ld.add_sequential_ids(df)

    print('Total of ', len(df), ' documents found in ', str(data_dir))

    df = cd.parallell_apply(df, cd.apply_text_preprocessing,
                        target_column = 'raw_text',
                        new_column_name = 'clean_text')

    df = cd.parallell_apply(df, cd.apply_convert_to_doc,
                        target_column = 'clean_text',
                        new_column_name = 'doc')

    df['sentences'] = df.apply(lambda row: cd.\
                    apply_get_sentence_list(row, 'doc'), axis=1)

    df['clean_sentences'] = df.apply(lambda row: cd.\
                    apply_clean_sentence_list(row, 'sentences'), axis=1)

    sd.save_sentences_to_file('all_sentences', df,
                            clean_sentence_column='clean_sentences',
                            group_size=3)

    df = sd.drop_columns(df, drop_columns=['doc', 'sentences'])
    sd.save_sentence_metadata('metadata', df)

    # print(df.head())
