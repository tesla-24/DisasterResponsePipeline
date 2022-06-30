# import libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Loads the cleaned data.
    Parameters:
        database_filepath: str
    Returns:
        X: dataframe
        Y: target dataframe
        category_names: list of names of categories
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    #Y = Y.drop(['child_alone'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    ''' Normalize, remove stopwords, stem the words in the given text string and return tokens'''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
    return stemmed_tokens


def build_model():
    '''Returns a model that classifies the given data into 36 categories'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'vect__max_features': [None, 10, 50],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [10, 30, 50]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the classification report for each category
    Parameters:
        model: Pipeline
            ML model to be used for classification
        X_test: pandas Dataframe
            training dataframe
        Y_test: pandas Dataframe
            target dataframe
        category_names: list
            names of the target classes
    '''
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Saves the model at the given path
    Parameters:
        model: Pipeline
            model to be saved
        model_filepath: str
            path at which model will be saved
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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