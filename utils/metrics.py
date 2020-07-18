import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix

def remove_padding(batch_data, lengths):
    ans = []
    for i in range(batch_data.shape[0]):
        ans.append(batch_data[i, :lengths[i]])
    return ans

def scratch_data(data_lst):
    data = np.concatenate(data_lst)
    return data

def evaluate_regression(y_true, y_pred):
    """ Evaluate the regression performance
        Params:
        y_true, y_pred: np.array()
        Returns:
        mse, rmse, pcc, ccc
    """
    assert y_true.ndim==1 and y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(y_true, y_pred)[0][1]
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    ccc = 2 * np.cov(y_true, y_pred, ddof=0)[0][1] / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    return mse, rmse, pcc, ccc

def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, recall, f1, cm, .66 * f1 + .34 * recall