import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    df_merged = pd.merge(df_messages, df_categories, on='id')

    categories_new = df_merged['categories'].str.split(';', n=36, expand=True)

    # select the first row of the categories dataframe
    row =list(categories_new.iloc[0])
    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for i in range(len(row)):
        category_colnames.append(row[i].split('-')[0])

    categories_new.columns = category_colnames

    for column in categories_new:
        # set each value to be the last character of the string
        categories_new[column] = categories_new[column].astype(str).str.split('-').str.get(1)
        # convert column from string to numeric
        categories_new[column] = pd.to_numeric(categories_new[column])

    df_merged.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df_merged, categories_new], axis=1)

    return df

def clean_data(df):

    #drop duplicates
    df.drop_duplicates(subset ="id",keep = False, inplace = True)

    df.drop_duplicates(subset ="message",keep = False, inplace = True)

    return df

def save_data(df, database_filename):

    df=df

    conn = sqlite3.connect(database_filename)

    df.to_sql('MessCat', con = conn, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
