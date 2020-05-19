import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
import sqlite3
import pickle

def load_data(database_filepath):

    conn = sqlite3.connect(database_filepath)

    df_ml = pd.read_sql("SELECT * FROM MessCat", con=conn)

    X =df_ml.message.values
    Y =df_ml.drop(['id','message','original','genre'], axis=1)

    category_names=Y.columns

    Y=Y.values

    return X, Y, category_names


def tokenize(text):

    stop_words = stopwords.words("english")

    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    pipeline =  Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('multi_target_forest',MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1) )
    ])

    # specify parameters for grid search
    parameters = {
    'multi_target_forest__estimator__n_estimators': [10],
    'multi_target_forest__estimator__min_samples_split': [2],
    'tfidf__use_idf': [False],
    'vect__max_df': [1.0],
    'vect__max_features': [5000],
    'vect__ngram_range': [(1, 2)]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        cr=classification_report(Y_test[:,i],Y_pred[:,i])
        print ("{}\n {}".format(category_names[i], cr))

    #print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):

    # Save to file in the curren working directory
    pkl_filename = model_filepath
    #/dosyayı/oluşturmak/istediğimiz/dizin/dosya_adı

    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    # Load from file
    with open(pkl_filename, 'rb') as file:
        MessCat_model = pickle.load(file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
