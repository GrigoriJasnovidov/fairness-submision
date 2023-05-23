import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def cuae(y_true, y_pred, sensitive_features):
    '''
    y_true - stands for the true label
    y_pred - a forecast
    sensitive_features - sensitive attribute
    '''
    true = np.array(y_true)
    pred = np.array(y_pred)
    protected = np.array(sensitive_features)
    df = pd.DataFrame({'true': true, 'pred': pred, 'protected': protected}).astype('category')
    classes = df['true'].drop_duplicates()
    protected_groups_values = df['protected'].drop_duplicates()
    np_ans = np.zeros(shape=[len(protected_groups_values), len(classes)])
    for j in range(len(protected_groups_values)):
        for i in range(len(classes)):
            protected_value = protected_groups_values[protected_groups_values.index[j]]
            current_part = df[df['protected'] == protected_value]
            ndf = current_part[(current_part['true'] == classes[classes.index[i]])]
            res = accuracy_score(ndf['true'], ndf['pred'])
            np_ans[j, i] = res
    df = pd.DataFrame(np_ans, columns=np.array(classes), index=np.array(protected_groups_values))

    max_diff = []
    max_ratio = []

    for i in df.columns:
        column = df[i]
        sort = np.array(column.sort_values())
        max_ratio.append(sort[-1] / sort[0])
        max_diff.append(sort[-1] - sort[0])
    max_diff = np.array(max_diff)
    max_ratio = np.array(max_ratio)
    total_diff = max_diff.max()
    total_ratio = max_ratio.max()
    global_max = df.max().max()
    global_min = df.min().min()
    variation = global_max - global_min

    ans = {'df': df,
           'diff': total_diff,
           'ratio': total_ratio,
           'variation': variation}
    return ans
