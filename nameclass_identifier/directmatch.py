import pandas as pd
import os
import numpy as np

def dmatch(df, fname_col, lname_col, source_fname, source_lname):
    """
    Based on a source dataframe, calculate the proportion of MENA and Arab populations,
    perform string stripping, drop NA values.

    Args:
        df: Source dataframe
        fname_col: name of the column which contains the first names
        lname_col: name of the column which contains the last names
        source_fname: source file or dataframe from which the first names are read from
        source_lname: source file or dataframe from which the last names are read from

    Returns:
        match: cleaned up original df dataframe merged with the dataframe which contains the first names,
        last names, and the proportion of Arab and MENA populations.

    """
    # Load source names list
    print("loading the source first names and last names")
    if isinstance(source_fname, str) and os.path.isfile(source_fname):
        fnames = pd.read_csv(source_fname,
                         names = ["name", "arab", "nonarab", "sasian", "others", "total"])
    elif isinstance(source_fname, pd.core.frame.DataFrame):
        fnames = source_fname

    if isinstance(source_lname, str) and os.path.isfile(source_lname):
        lnames = pd.read_csv(source_lname,
                             names=["name", "arab", "nonarab", "sasian", "others", "total"])
    elif isinstance(source_lname, pd.core.frame.DataFrame):
        lnames = source_lname


    # Calculate proportion Arab and MENA
    print("Calculting proportions of Arab and MENA for first names and last names")
    fnames['p_arab_first'] = fnames['arab'] / fnames['total']
    fnames['p_mena_first'] = (fnames['arab'] + fnames['nonarab']) / fnames['total']
    lnames['p_arab_last'] = lnames['arab'] / lnames['total']
    lnames['p_mena_last'] = (lnames['arab'] + lnames['nonarab']) / lnames['total']

    #Drop NAs
    print("Dropping NA values from the dataframe, subsetting based on fname_col, lname_col")
    df = df.replace('NaN', np.NaN)
    df.dropna(subset=[fname_col, lname_col], inplace=True)
    # Strip white space from names
    print("Stripping white spaces from the dataframe, subsetting based on fname_col, lname_col")
    df[fname_col] = df[fname_col].str.strip()
    df[lname_col] = df[lname_col].str.strip()

    print("Merging fnames and lnames with the original df dataframe object")
    match_first = df.merge(fnames, left_on=fname_col, right_on = 'name', how='left')
    match = match_first.merge(lnames, left_on=lname_col, right_on = 'name', how='left')
    return(match)
