import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


def evaluate_classifier():
    with open('classifier_model.pkl', 'rb') as model_handle:
        loaded_model = pickle.load(model_handle)
    
    test_dataset = pd.read_csv('test/processed_testing.csv')
    
    outcome_label = 'outcome'
    y_true = test_dataset[outcome_label]
    x_test = test_dataset.drop(columns=[outcome_label])
    
    predictions = loaded_model.predict(x_test)
    performance_metric = accuracy_score(y_true, predictions)
    
    print(f'Model test accuracy is: {performance_metric:.3f}')


evaluate_classifier()