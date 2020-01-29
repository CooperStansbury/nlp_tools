# Command Line Tools
This directory is a number of ad hoc tools used for exploratory data analysis/data structure extraction. This tools are not generally robust.

1. **dir2csv.py** : returns a new .csv file with various useful fields for downstream processing.
    - `--dir` a directory of .txt files. Encoding not handled.
    - `--dest` is the name and destination of the output file.

1. **dir2sent.py** : returns a new .csv file with sentences from an input directory.
    - `--dir` a directory of .txt files. Encoding handled, light data cleaning.
    - `--model` the `spaCy` language to use, defaults to `en_core_web_sm`.
    - `--dest` is the name and destination of the output file.

1. **form_classifier.py**: generate a .csv file containing results for a number of sklearn models on the input data.
    - `--src`: the file to perform predictions on. Currently, this tool requires the output of wordcount_matrix.py
    - `--dest` is the name and destination of the output file.

1. **n_gram_counter.py**: to generate a .csv file containing n-gram counts for a single input file.
    - `--gram_size` the maximum number of grams to generate. All n-grams from 0 - gram_size will be generated.
    - `--input_file` is the file to process.

1. **named_entity_extractor.py**: to print untrained named entities from an input file using the python module `spaCy`.
    - `--input` the file to extract named entities from.

1. **sentence_extractor.py**: to print untrained named entities from an input file using the python module `spaCy`.
    - `--src` the file to run on.
    - `--model` the `spaCy` language to use, defaults to `en_core_web_sm`.

1. **tf-idf.py**: generate a .csv file containing a list of features computed using TF-IDF for each document in the input directory.
    - `--src` is the root directory for the extraction.
    - `--file_type` is the file type, specified as a string value corresponding to a valid extension (ex. '.txt').
    - `--dest` is the name and destination of the output file. Example below:

1. **wordcount_matrix**: generate a .csv file containing high-level data for all files in a specified directory.
    - `--src` is the root directory for the search.
    - `--file_type` is the file type, specified as a string value corresponding to a valid extension (ex. '.txt').
    - `--keys` is an input `.csv` file containing search phrases, whose occurrences will be counted for each file in (1).
    - `--dest` is the name and destination of the output file.
