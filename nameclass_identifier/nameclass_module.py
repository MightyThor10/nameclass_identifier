#####################################################
############ Name Classifier ########################
####### Inputs: - list of names with country of origin data
#######         - columns with first and last names
####### Output: trained model
######################################################

from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as np_utils
import io
import sys
import os
from keras import callbacks
import re
import os
######################################################

########### Function to import data ################


def import_data(data, first_name, last_name, min_length=1, sep=",", miss_val='NaN', strip_drop = 'drop'):
    """
        Import data directly from a dataframe or a csv file and perform some data cleanup.

        Remove NaN rows, strip away whitespaces from names, turn them into lowercase, subset the df
        based on a minimum character limit for names, and give the option to drop rows with or strip
        non-alphabetic characters from names.

        Args:
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
    """
    # Read source list of names into a data frame
    if isinstance(data, str) and os.path.isfile(data):
        print("Creating a dataframe from the input csv")
        df = pd.read_csv(data, sep=sep, index_col=False)
    elif isinstance(data, pd.core.frame.DataFrame):
        print("input is already a dataframe.... Continuing")
        df = data
    else:
        # else just transform whatever the data is to a dataframe
        try:
            df = pd.DataFrame(data)
        except Exception:
            print("Transforming the data to a dataframe failed.... Make sure the data is compatible")
            return

    # Handle missing values
    df = df.replace(miss_val, np.NaN)
    print("First names loaded: " + str(len(df[first_name])))
    print("Number of missing values to drop in first name: " + str(df[first_name].isna().sum()))
    print("Last names loaded: " + str(len(df[last_name])))
    print("Number of missing values to drop in last name: " + str(df[last_name].isna().sum()))
    df.dropna(subset=[first_name, last_name], inplace=True)

    # Handle rows with non-alphabetic characters: either drop or strip
    print("Number of rows with non-alphabetic characters in first name: " + str((~df[first_name].str.isalpha()).sum()))
    print("Number of rows with non-alphabetic characters in last name: " + str((~df[last_name].str.isalpha()).sum()))
    if strip_drop == 'strip':
        print("Stripping non alphabetic values")
        pattern = re.compile('[^A-Za-z]')
        df[first_name] = df[first_name].str.replace(pattern, '')
        df[last_name] = df[last_name].str.replace(pattern, '')
    else:
        print("Dropping non alphabetic values")
        df = df.drop(df[~df[first_name].str.isalpha()].index)
        df = df.drop(df[~df[last_name].str.isalpha()].index)

    # Strip whitespaces in names:
    df[first_name] = df[first_name].str.replace(" ", "")
    df[last_name] = df[last_name].str.replace(" ", "")

    # Covert all names to lowercase
    df[first_name] = df[first_name].str.lower()
    df[last_name] = df[last_name].str.lower()

    # Subset data frame (optional)
    df = df[df[first_name].str.len() >= min_length]
    df = df[df[last_name].str.len() >= min_length]

    print("First names ready: " + str(len(df[first_name])))
    print("Last names ready: " + str(len(df[last_name])))
    # Return the data frame

    # This message will be printed if the function ran successfully
    print('data import successful')

    return df



##################################################
def find_ngrams(vocab, text, n):
    """
    Find and return list of the index of n-grams in the vocabulary list.
    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.
    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams
    Returns:
        list: List of the index of n-grams in the vocabulary list.
    """
    wi = []
    if not isinstance(text, str):
        return wi
    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = ''.join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi

########## Function to create model ###############
def run_model(sdf, NGRAMS, min_length, batch_size, epochs, model_name, first_name, last_name, group_var):
    """
    create a prediction model using the source dataframe sdf, return the trained model, grouped
    dataframe based on the grouping variable, classification report, and the generated vocabulary list.
    Args:
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
    """
    # Drop if names are shorter than minimum length
    print("Dropping names shorter than the minimum length\n")
    sdf = sdf[sdf[first_name].str.len() >= min_length]
    sdf = sdf[sdf[last_name].str.len() >= min_length]
    # Return first and last name as a string in which the first letter of each name is uppercase and the following letters are lowercase
    print("Capitalizing first letter of all names")
    sdf[first_name] = sdf[first_name].str.title()
    sdf[last_name] = sdf[last_name].str.title()
    # Group the data frame by race and aggregating by first name
    print("Grouping the data by the passed group_var and aggregating by first name")
    sdf.groupby(group_var).agg({first_name: 'count'})

    # concatenate last name and first name
    sdf['name_last_name_first'] = sdf[last_name] + ' ' + sdf[first_name]

    # Output #1: Groups
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    groups = pd.DataFrame(sdf.groupby(group_var).agg({first_name: 'count'}).index.get_level_values(0))
    groups.to_csv(os.path.join(model_name, 'groups.csv'))

    # build n-gram list
    print("Building the n-gram list\n")
    vect = CountVectorizer(analyzer='char', max_df=0.3, min_df=3, ngram_range=(NGRAMS, NGRAMS), lowercase=False)
    # transform names into vectors
    a = vect.fit_transform(sdf.name_last_name_first)
    # returns dictionary where keys are ngrams and values are number of occurrences
    vocab = vect.vocabulary_

    # sort n-gram by freq (highest -> lowest)
    words = []
    for b in vocab:
        c = vocab[b]
        words.append((a[:, c].sum(), b))
    words = sorted(words, reverse=True)
    words_list = [w[1] for w in words]
    num_words = len(words_list)
    print("num_words = %d" % num_words)

    # build X from index of n-gram sequence
    X = np.array(sdf.name_last_name_first.apply(lambda c: find_ngrams(words_list, c, NGRAMS)))

    # check max/avg feature
    X_len = []
    for x in X:
        X_len.append(len(x))

    # find the maximum and average name lengths
    max_feature_len = max(X_len)
    avg_feature_len = int(np.mean(X_len))

    print("Max feature len = %d, Avg. feature len = %d" % (max_feature_len, avg_feature_len))
    # maps categories to numbers
    y = np.array(sdf[group_var].astype('category').cat.codes)

    # Split train and test dataset
    print("Splitting data into train and test dataset........................................................\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    feature_len = 25  # avg_feature_len # cut texts after this number of words (among top max_features most common words)
    batch_size = batch_size

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    # transforms list of sequences into a 2D numpy array,
    # pads and truncates sequences so they are the same length
    X_train = sequence.pad_sequences(X_train, maxlen=feature_len)
    X_test = sequence.pad_sequences(X_test, maxlen=feature_len)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # figre out number of classes
    num_classes = np.max(y_train) + 1
    print(num_classes, 'classes')
    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')
    # convert vector to matrix
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    ##### Build model #####
    print('Building model...................................................\n')
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=feature_len))
    model.add(LSTM(128, activation = 'tanh', dropout=0.2, recurrent_dropout=0))
    model.add(Dense(num_classes, activation='softmax'))

    ##### Compile model #####
    print("Compiling the model.............................................\n")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    ##########################

    print(model.summary())

    ##### Train model #####
    print('Training..........................................................\n')
    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=5,
                                            restore_best_weights=True)
    print(model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1, verbose=2,
              callbacks =[earlystopping]))

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size, verbose=2)
    print('Test score:', score)
    print('Test accuracy:', acc)

    ######## Make predictions ################
    print("Making predictions using the created model........................................\n")
    y_pred = np.argmax(model.predict(X_test), axis=-1)





    # Output #2: Classification report
    p = model.predict(X_test, verbose=2)  # to predict probability
    target_names = list(sdf[group_var].astype('category').cat.categories)
    # print(classification_report(np.argmax(y_test_new, axis=1), y_pred, target_names=target_names))
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    ######## Create classification report for inputted parameters #############
    # x = (classification_report(np.argmax(y_test_new, axis=1), y_pred, target_names=target_names))
    x = (classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
    str = io.StringIO(x)
    df = pd.read_fwf(str)
    pivoted = df.pivot(values=['precision', 'recall', 'f1-score', 'support'], columns=['Unnamed: 0'])
    if df.index.nlevels == 1:
        df1 = pivoted.apply(lambda x: pd.Series(x.dropna().to_numpy())).iloc[[0]]
    df1.to_csv(os.path.join(model_name, 'classification_report.csv'), index=False, encoding='utf-8')
    print('successful')

    model.save(os.path.join(model_name, 'mod.h5'))
    # Output #3: Vocab/words
    words_df = pd.DataFrame(words_list, columns=['vocab'])
    words_df.to_csv(os.path.join(model_name, 'words_df.csv'), index=False, encoding='utf-8')

    return groups, df1, words_df, model


#######################################################################

def pred_name(df, lname_col, fname_col, ngrams, model_name, feature_len):
    """Predict the race/ethnicity by the full name.
    Args:
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
    """
    if lname_col not in df.columns:
        print("No column `{0!s}` in the DataFrame".format(lname_col))
        return df
    if fname_col not in df.columns:
        print("No column `{0!s}` in the DataFrame".format(fname_col))
        return df

    if not os.path.exists(model_name):
        print("Cannot find the model name folder provided")

    df['__name'] = (df[lname_col].str.strip() + ' ' + df[fname_col].str.strip()).str.title()

    #  sort n-gram by freq (highest -> lowest)
    vocab = pd.read_csv(os.path.join(model_name, 'words_df.csv'))
    vocab = list(vocab.iloc[:, 0])
    groups = pd.read_csv(os.path.join(model_name, 'groups.csv'))

    # model = load_model(MODEL)
    print("Loading the desired model")
    model = load_model(os.path.join(model_name, 'mod.h5'))

    # build X from index of n-gram sequence
    print("Building X from index of the n-gram sequenece")
    X = np.array(df.__name.apply(lambda c: find_ngrams(vocab, c, n=ngrams)))
    X = sequence.pad_sequences(X, maxlen=feature_len)
    print(X)

    print("Predicting X using the loaded model")
    predict_x = model.predict(X)
    print(predict_x)
    pdf = pd.DataFrame.from_records(
        predict_x, columns=groups.iloc[:,1])
    pdf['prediction'] = pdf.idxmax(axis=1)

    rdf = pd.concat([df.reset_index(drop=True), pdf], axis=1)
    rdf = rdf.add_prefix(model_name + '_')
    #rdf.to_csv(os.path.join(model_name, 'predicted.csv'), index = False, encoding = 'utf-8')
    print(rdf)
    return rdf

def test():
    print("Testing if nameclass_module works")
