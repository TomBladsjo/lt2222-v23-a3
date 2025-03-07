import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def readfile(filename):
    """Takes as input the name of a file and returns the content as a string,
    cleaned from email headers, forwarded messages etc.
    """
    document = ''
    main_body_start = False
    with open(filename) as f:
        for line in f:
            text = line.replace('\n', ' ') 
            if text.lstrip(' ')[:5] == '-----':
                # Removes last two words of email which are usually the sender's name.
                return document.rstrip().rsplit(maxsplit=2)[0] if not document.isspace() else ''
            if main_body_start == True:
                document += text
            elif text[:10] == 'X-FileName':
                main_body_start = True
    # Again, removes what might be the sender's name. This is in no way a perfect solution.
    return document.rstrip().rsplit(maxsplit=2)[0] if not document.isspace() else ''

def read_data(rootdir):
    """Takes as input the path to a directory containing multiple subdirectories,
    and returns two lists of equal length: one of the (cleaned) files as strings, 
    and one of the names of the subdirectories where each file was found.
    """
    alldocs = []
    allauthors = []
    for author in os.listdir(path=rootdir):
        alldocs.extend(readfile(rootdir+author+'/'+file) for file in os.listdir(path=rootdir+author))  
        allauthors.extend(author for file in os.listdir(path=rootdir+author))
    return alldocs, allauthors


def feature_table(documents, labels, dims, testsize):
    """Turns a list of documents and a list of labels into a pandas DataFrame of
    dimensionality-reduced vectorized representations.
    
    Args: 
        documents: A list of documents as strings.
        labels: A list of the corresponding labels as strings.
        dims: The dimensions of the output representation.
        testsize: The percentage of instances to be labelled as test.
    """
    vectorizer = CountVectorizer(max_features=dims)
    X = vectorizer.fit_transform(documents).toarray()

    docsdf = pd.DataFrame(X)
    docsdf['class'] = labels
    classes = list(docsdf['class'].unique())
    docsdf['class_id'] = docsdf['class'].map(lambda x: classes.index(x))

    grouped = docsdf.groupby("class", group_keys=False)
    testdf = grouped.apply(lambda x: x.sample(frac=testsize/100))
    testdf['split'] = 'test'
    traindf = docsdf.drop(testdf.index, axis=0)
    traindf['split'] = 'train'

    fulldf = pd.concat([traindf, testdf], ignore_index=True)
    fulldf.index.name = 'vectors'

    return fulldf




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    docs, authors = read_data(args.inputdir)


    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    table = feature_table(docs, authors, args.dims, args.testsize)


    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    table.to_csv(args.outputfile)

    print("Done!")
    















