import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


def build_classifier():
    training_set = pd.read_csv('train/processed_training.csv')
    
    target_var = 'outcome'
    y_training = training_set[target_var]
    x_training = training_set.drop(columns=[target_var])
    
    classifier = LogisticRegression(max_iter=150, random_state=42)
    classifier.fit(x_training, y_training)
    
    with open('classifier_model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)


build_classifier()