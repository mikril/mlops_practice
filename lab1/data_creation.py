import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification


def generate_dataset():
    X_raw, y_raw = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=123
    )
    
    feature_names = [f'param_{i+1}' for i in range(X_raw.shape[1])]
    dataset_df = pd.DataFrame(X_raw, columns=feature_names)
    dataset_df['outcome'] = y_raw
    
    sample_count = dataset_df.shape[0]
    test_sample_size = int(sample_count * 0.3)
    
    test_indices = np.random.RandomState(123).choice(
        sample_count, 
        size=test_sample_size, 
        replace=False
    )
    
    test_subset = dataset_df.iloc[test_indices].copy().reset_index(drop=True)
    train_subset = dataset_df.drop(index=test_indices).reset_index(drop=True)
    
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    
    train_subset.to_csv('train/training_data.csv', index=False)
    test_subset.to_csv('test/testing_data.csv', index=False)


if __name__ == '__main__':
    generate_dataset()