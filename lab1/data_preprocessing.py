import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def process_data():
    train_source = pd.read_csv('train/training_data.csv')
    test_source = pd.read_csv('test/testing_data.csv')
    
    outcome_column = 'outcome'
    y_train_part = train_source[outcome_column]
    x_train_part = train_source.drop(columns=[outcome_column])
    y_test_part = test_source[outcome_column]
    x_test_part = test_source.drop(columns=[outcome_column])
    
    normalizer = StandardScaler()
    x_train_normalized = normalizer.fit_transform(x_train_part.values)
    x_test_normalized = normalizer.transform(x_test_part.values)
    
    param_labels = [f'param_{j+1}' for j in range(x_train_normalized.shape[1])]
    x_train_frame = pd.DataFrame(x_train_normalized, columns=param_labels)
    x_test_frame = pd.DataFrame(x_test_normalized, columns=param_labels)
    
    train_processed = x_train_frame.copy()
    train_processed[outcome_column] = y_train_part
    test_processed = x_test_frame.copy()
    test_processed[outcome_column] = y_test_part
    
    train_processed.to_csv('train/processed_training.csv', index=False)
    test_processed.to_csv('test/processed_testing.csv', index=False)


process_data()