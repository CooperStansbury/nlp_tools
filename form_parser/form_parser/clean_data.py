"""
description: functions to perform various data preprocessing steps on textual
data

for more information: the following citation is for the language library
developed by spaCy authors:

    [1] Models for the spaCy Natural Language Processing (NLP) library:
    explosion/spacy-models. Explosion, 2019.

"English multi-task CNN trained on OntoNotes, with GloVe vectors trained
on Common Crawl. Assigns word vectors, context-specific token vectors,
POS tags, dependency parse and named entities. This is what allows us
to predict..."
"""

# non-local imports
import re
import spacy
import dask.dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count

# load once, call many
nlp_larg = spacy.load('en_core_web_lg')

## TODO: this is a TON of overhead for a simple method call to the parser
## I think next refactoring should stip the instantiation of the entire
## spacy.token.span object (doc), if possible and use only the parser.
## Should consider Stanford NLP for this task.


def encode_string(string):
    """ return new encoded string ascii """
    return string.encode(encoding = 'ascii', errors = 'replace')


def decode_string(string):
    """ return new decoded string ascii """
    return string.decode(encoding='ascii',errors='strict')


def encoding_cleanup(string):
    """ strip encoding error characters and return new string """
    return str(string).replace("?", " ")


def strip_and_lower_string(string):
    """ return new string with lowercase characters and strip whitespace """
    return re.sub(' +', ' ', str(string).lower().strip())


def strip_special_chars_string(string):
    """ return a new sting without special characters """
    return re.sub('[^A-Za-z0-9]+', ' ', string).strip()


def remove_signature_lines(string):
    """ strip characters that are likely signature lines """
    return string.replace("_", "")


def apply_text_preprocessing(row, target_column):
    """ perform text processing on raw data to new field """
    force_encoding = decode_string(encode_string(row[target_column]))
    clean_string = strip_and_lower_string(encoding_cleanup(force_encoding))
    return remove_signature_lines(clean_string)


def apply_convert_to_doc(row, target_column):
    """ return spacy doc object from a text field """
    spacy_doc = nlp_larg(str(row[target_column]))
    return spacy_doc


def apply_get_sentence_list(row, target_column):
    """ return list of sentences from doc object field;
    each item will be token span """
    return list(row[target_column].sents)


def apply_clean_sentence_list(row, target_column):
    """  clean sentence list, return new column """
    cleaned_sentence_list = []
    for sentence in row[target_column]:
        cleaned_sentence_list.append(strip_special_chars_string(sentence.text))
    return cleaned_sentence_list


def parallell_apply(df, apply_function, target_column, new_column_name):
    """ return new pd.DataFrame with new column of cleaned text data """
    print('Creating new column: ', str(new_column_name))
    df[new_column_name] = dd.from_pandas(df,npartitions=cpu_count()).\
        map_partitions(lambda df : df.apply(
            lambda x : apply_function(x, target_column),axis=1)).\
                compute(scheduler='threads')
    print('Return DataFrame with columns: ', str(df.columns))
    return df
