import pandas as pd
import numpy as np

def one_hot_to_index(y):
    return np.argmax(y, axis=1) + 1

def filter_others_data_for_X(y, X, vals):
    X = np.array(X)
    y = np.array(y)
    mask = ~np.isin(y, vals)
    return X[mask], y[mask]

def filter_data_for_X(y, X, vals):
    X = np.array(X)
    y = np.array(y)
    mask = np.isin(y, vals)
    return X[mask], y[mask]

def transform_labels_model(y, ori_val, new_val):
    for i in range(len(ori_val)):
        y = np.where(y == ori_val[i], new_val[i], y)
    return y

def save_filtered_data(X, y, X_filename, y_filename):
    pd.DataFrame(X).to_csv(X_filename, index=False)
    pd.DataFrame(y).to_csv(y_filename, index=False)
    
def to_one_hot(labels, num_classes=7):
    return np.eye(num_classes)[labels - 1]

# def transform_labels_model1(y):
#     y_transformed = np.where((y != 4) & (y != 5), 0, y)
#     y_transformed = np.where(y_transformed == 4, 1, y_transformed)
#     y_transformed = np.where(y_transformed == 5, 2, y_transformed)
#     return y_transformed