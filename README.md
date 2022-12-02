# <p align="center"> Welcome to Nameclass Identifier </p>

## <p align="center"> Documentation </p>

## Scripts that can be imported and used, along with necessary parameters and appropriate value types:

#### 1. nameclass_module
Functions available:

```
import_data(data, first_name, last_name, min_length=1, sep=",", miss_val='NaN', strip_drop = 'drop')

        Description: Import data directly from a dataframe or a csv file and perform some data cleanup. Remove NaN rows, strip away whitespaces from names, 
        turn them into lowercase, subset the df based on a minimum character limit for names, and give the option to drop rows with or strip non-alphabetic characters from names.

        Argumentss:
            data (df object, string): a dataframe object or a string containing the relative or absolute
            path for a csv file.
            first_name, last_name (string): names of the columns containing the first names and last names
            min_length (int, default=1): minimum length of characters first names and last names should have
            sep (string, default=", "): character, i.e. comma, space, etc. on which the csv file should be read with.
            miss_val (string, default = "NaN"): Missing character in the columns of the dataframe (to be replaced by np.NaN)
            strip_drop (string, default = "drop"): Option to choose either 'drop' to drop rows with non-alphabetic characters or 'strip'
            to strip away the non-alphabetic characters.

        Returns:
            df object: Cleaned-up dataframe.

```
```
find_ngrams(vocab, text, n)
    
    Description: Find and return list of the index of n-grams in the vocabulary list.
    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.
    
    Argumentss:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams
        
    Returns:
        list: List of the index of n-grams in the vocabulary list.
```

```
run_model(sdf, NGRAMS, min_length, batch_size, epochs, model_name, first_name, last_name, group_var)

    Description: create a prediction model using the source dataframe sdf, return the trained model, grouped
    dataframe based on the grouping variable, classification report, and the generated vocabulary list.
    
    Arguments:
        sdf: (df object) Source dataframe
        NGRAMS: (int) N-grams
        min_length: (int) min length for the names columns
        batch_size: (int) batch size on which the model should train/fit on
        epochs: (int) epochs size on which the model should train/fit on
        model_name: (str) desired name for the model
        first_name: (str) name of the column that holds first names
        last_name: (str) name of the column that holds last names
        group_var: (str) name of the variable on which grouping should be done

    Returns:
        model: (df object) model that has been trained and ready for prediction
        groups: (df object) created from sdf but grouped based on the grouping variable and with
        aggregation of first names count
        df1: (df object) dataframe containing the final classification report for the model
        words_df: (df object) dataframe containing all the words in the vocabulary list
```

```
pred_name(df, lname_col, fname_col, ngrams, model_name, feature_len)

    Description: Predict the race/ethnicity by the full name.
    
    Arguments:
        df (:obj:`DataFrame`): Pandas DataFrame containing the last name and
            first name column.
        lname_col (str or int): Column's name or location of the last name in
            DataFrame.
        fname_col (str or int): Column's name or location of the first name in
            DataFrame.
        ngrams: (int) N-grams
        model_name: (str) path (absolute or relative) to the folder where the prediction model exists
        feature_len: (int) maximum length of all sequences. If not provided, sequences will be padded to the
            length of the longest individual sequence.
            
    Returns:
        DataFrame: Pandas DataFrame with additional columns:
            - `race` the predict result
            - Additional columns for probability of each classes.
```
<br></br>

#### 2. directmatch.py

Functions available:

```
 dmatch(df, fname_col, lname_col, source_fname, source_lname)
    
    Description: Based on a source dataframe, calculate the proportion of MENA and Arab populations,
    perform string stripping, drop NA values.

    Arguments:
        df: Source dataframe
        fname_col: name of the column which contains the first names
        lname_col: name of the column which contains the last names
        source_fname: source file or dataframe from which the first names are read from
        source_lname: source file or dataframe from which the last names are read from

    Returns:
        match: cleaned up original df dataframe merged with the dataframe which contains the first names,
        last names, and the proportion of Arab and MENA populations.
```


## Guide to packages and dependencies used:
Dependencies and Packages used and/or to be installed before usage: 
* pandas
* os
* numpy
* sys
* from sklearn.feature_extraction.text import CountVectorizer
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import classification_report, confusion_matrix
* keras 
* from tensorflow.keras.preprocessing import sequence 
* from tensorflow.keras.models import Sequential 
* from tensorflow.keras.layers import Dense, Embedding, Dropout, Activation 
* from tensorflow.keras.layers import LSTM 
* from tensorflow.keras.models import load_model 
* from tensorflow.keras import utils as np_utils
* import io
* from keras import callbacks
* import re

### License

Copyright (c) 2022, ElyasB

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.