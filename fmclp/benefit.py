import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def benefit(res):
    classifier = res['model']['estimator']
    features = res['model']['y_test']
    data = features.copy()
    data['label'] = res['model']['x_test']
    data['prediction'] = classifier.predict(features)
    data['fair_prediction'] = res['predictions']['preds']
    data = data[['attr', 'label', 'prediction', 'fair_prediction']]

    # unfair_preds

    unfair_data_1 = data[data['attr'] == 1].drop(['attr', 'fair_prediction'], axis=1)
    unfair_data_0 = data[data['attr'] == 0].drop(['attr', 'fair_prediction'], axis=1)

    unfair_1_accuracy = accuracy_score(unfair_data_1.label, unfair_data_1.prediction)
    unfair_0_accuracy = accuracy_score(unfair_data_0.label, unfair_data_0.prediction)

    unfair_discriminated_1_2 = unfair_data_1[unfair_data_1['label'] == 2]
    unfair_discriminated_1_21 = unfair_discriminated_1_2[unfair_discriminated_1_2['prediction'] == 1]
    unfair_discriminated_1_20 = unfair_discriminated_1_2[unfair_discriminated_1_2['prediction'] == 0]
    unfair_discriminated_1_1 = unfair_data_1[unfair_data_1['label'] == 1]
    unfair_discriminated_1_10 = unfair_discriminated_1_1[unfair_discriminated_1_1['prediction'] == 0]

    unfair_discriminated_0_2 = unfair_data_0[unfair_data_0['label'] == 2]
    unfair_discriminated_0_21 = unfair_discriminated_0_2[unfair_discriminated_0_2['prediction'] == 1]
    unfair_discriminated_0_20 = unfair_discriminated_0_2[unfair_discriminated_0_2['prediction'] == 0]
    unfair_discriminated_0_1 = unfair_data_0[unfair_data_0['label'] == 1]
    unfair_discriminated_0_10 = unfair_discriminated_0_1[unfair_discriminated_0_1['prediction'] == 0]

    unfair_privileged_1_0 = unfair_data_1[unfair_data_1['label'] == 0]
    unfair_privileged_1_01 = unfair_privileged_1_0[unfair_privileged_1_0['prediction'] == 1]
    unfair_privileged_1_02 = unfair_privileged_1_0[unfair_privileged_1_0['prediction'] == 2]
    unfair_privileged_1_1 = unfair_data_1[unfair_data_1['label'] == 1]
    unfair_privileged_1_12 = unfair_privileged_1_1[unfair_privileged_1_1['prediction'] == 2]

    unfair_privileged_0_0 = unfair_data_0[unfair_data_0['label'] == 0]
    unfair_privileged_0_01 = unfair_privileged_0_0[unfair_privileged_0_0['prediction'] == 1]
    unfair_privileged_0_02 = unfair_privileged_0_0[unfair_privileged_0_0['prediction'] == 2]
    unfair_privileged_0_1 = unfair_data_0[unfair_data_0['label'] == 1]
    unfair_privileged_0_12 = unfair_privileged_0_1[unfair_privileged_0_1['prediction'] == 2]

    unfair_l_1 = len(unfair_data_1)
    unfair_l_0 = len(unfair_data_0)

    udw21 = unfair_discriminated_1_21.shape[0] / unfair_l_1
    udw20 = unfair_discriminated_1_20.shape[0] / unfair_l_1
    udw10 = unfair_discriminated_1_10.shape[0] / unfair_l_1

    upw01 = unfair_privileged_1_01.shape[0] / unfair_l_1
    upw02 = unfair_privileged_1_02.shape[0] / unfair_l_1
    upw12 = unfair_privileged_1_12.shape[0] / unfair_l_1

    udo21 = unfair_discriminated_0_21.shape[0] / unfair_l_0
    udo20 = unfair_discriminated_0_20.shape[0] / unfair_l_0
    udo10 = unfair_discriminated_0_10.shape[0] / unfair_l_0

    upo01 = unfair_privileged_0_01.shape[0] / unfair_l_0
    upo02 = unfair_privileged_0_02.shape[0] / unfair_l_0
    upo12 = unfair_privileged_0_12.shape[0] / unfair_l_0

    # fair predictions

    fair_data_1 = data[data['attr'] == 1].drop(['attr', 'prediction'], axis=1)
    fair_data_0 = data[data['attr'] == 0].drop(['attr', 'prediction'], axis=1)

    fair_1_accuracy = accuracy_score(fair_data_1.label, fair_data_1.fair_prediction)
    fair_0_accuracy = accuracy_score(fair_data_0.label, fair_data_0.fair_prediction)

    fair_discriminated_1_2 = fair_data_1[fair_data_1['label'] == 2]
    fair_discriminated_1_21 = fair_discriminated_1_2[fair_discriminated_1_2['fair_prediction'] == 1]
    fair_discriminated_1_20 = fair_discriminated_1_2[fair_discriminated_1_2['fair_prediction'] == 0]
    fair_discriminated_1_1 = fair_data_1[fair_data_1['label'] == 1]
    fair_discriminated_1_10 = fair_discriminated_1_1[fair_discriminated_1_1['fair_prediction'] == 0]

    fair_discriminated_0_2 = fair_data_0[fair_data_0['label'] == 2]
    fair_discriminated_0_21 = fair_discriminated_0_2[fair_discriminated_0_2['fair_prediction'] == 1]
    fair_discriminated_0_20 = fair_discriminated_0_2[fair_discriminated_0_2['fair_prediction'] == 0]
    fair_discriminated_0_1 = fair_data_0[fair_data_0['label'] == 1]
    fair_discriminated_0_10 = fair_discriminated_0_1[fair_discriminated_0_1['fair_prediction'] == 0]

    fair_priveleged_1_0 = fair_data_1[fair_data_1['label'] == 0]
    fair_priveleged_1_01 = fair_priveleged_1_0[fair_priveleged_1_0['fair_prediction'] == 1]
    fair_priveleged_1_02 = fair_priveleged_1_0[fair_priveleged_1_0['fair_prediction'] == 2]
    fair_priveleged_1_1 = fair_data_1[fair_data_1['label'] == 1]
    fair_priveleged_1_12 = fair_priveleged_1_1[fair_priveleged_1_1['fair_prediction'] == 2]

    fair_priveleged_0_0 = fair_data_0[fair_data_0['label'] == 0]
    fair_priveleged_0_01 = fair_priveleged_0_0[fair_priveleged_0_0['fair_prediction'] == 1]
    fair_priveleged_0_02 = fair_priveleged_0_0[fair_priveleged_0_0['fair_prediction'] == 2]
    fair_priveleged_0_1 = fair_data_0[fair_data_0['label'] == 1]
    fair_priveleged_0_12 = fair_priveleged_0_1[fair_priveleged_0_1['fair_prediction'] == 2]

    fair_l_1 = len(fair_data_1)
    fair_l_0 = len(fair_data_0)
    fdw21 = fair_discriminated_1_21.shape[0] / fair_l_1
    fdw20 = fair_discriminated_1_20.shape[0] / fair_l_1
    fdw10 = fair_discriminated_1_10.shape[0] / fair_l_1

    fdo21 = fair_discriminated_0_21.shape[0] / fair_l_0
    fdo20 = fair_discriminated_0_20.shape[0] / fair_l_0
    fdo10 = fair_discriminated_0_10.shape[0] / fair_l_0

    fpw01 = fair_priveleged_1_01.shape[0] / fair_l_1
    fpw02 = fair_priveleged_1_02.shape[0] / fair_l_1
    fpw12 = fair_priveleged_1_12.shape[0] / fair_l_1

    fpo01 = fair_priveleged_0_01.shape[0] / fair_l_0
    fpo02 = fair_priveleged_0_02.shape[0] / fair_l_0
    fpo12 = fair_priveleged_0_12.shape[0] / fair_l_0

    # true label axis 'x'
    # prediction label axis 'y'

    unfair_1_matrix = pd.DataFrame(columns=[0, 1, 2], index=[0, 1, 2])
    unfair_1_matrix[0][0] = ''
    unfair_1_matrix[1][1] = ''
    unfair_1_matrix[2][2] = ''
    unfair_1_matrix[0][1] = upw01
    unfair_1_matrix[0][2] = upw02
    unfair_1_matrix[1][2] = upw12
    unfair_1_matrix[1][0] = udw10
    unfair_1_matrix[2][0] = udw20
    unfair_1_matrix[2][1] = udw21

    unfair_0_matrix = pd.DataFrame(columns=[0, 1, 2], index=[0, 1, 2])
    unfair_0_matrix[0][0] = ''
    unfair_0_matrix[1][1] = ''
    unfair_0_matrix[2][2] = ''
    unfair_0_matrix[0][1] = upo01
    unfair_0_matrix[0][2] = upo02
    unfair_0_matrix[1][2] = upo12
    unfair_0_matrix[1][0] = udo10
    unfair_0_matrix[2][0] = udo20
    unfair_0_matrix[2][1] = udo21

    fair_1_matrix = pd.DataFrame(columns=[0, 1, 2], index=[0, 1, 2])
    fair_1_matrix[0][0] = ''
    fair_1_matrix[1][1] = ''
    fair_1_matrix[2][2] = ''
    fair_1_matrix[0][1] = fpw01
    fair_1_matrix[0][2] = fpw02
    fair_1_matrix[1][2] = fpw12
    fair_1_matrix[1][0] = fdw10
    fair_1_matrix[2][0] = fdw20
    fair_1_matrix[2][1] = fdw21

    fair_0_matrix = pd.DataFrame(columns=[0, 1, 2], index=[0, 1, 2])
    fair_0_matrix[0][0] = ''
    fair_0_matrix[1][1] = ''
    fair_0_matrix[2][2] = ''
    fair_0_matrix[0][1] = fpo01
    fair_0_matrix[0][2] = fpo02
    fair_0_matrix[1][2] = fpo12
    fair_0_matrix[1][0] = fdo10
    fair_0_matrix[2][0] = fdo20
    fair_0_matrix[2][1] = fdo21

    udo_vector = [udo10, udo20, udo21]
    udw_vector = [udw10, udw20, udw21]
    fdo_vector = [fdo10, fdo20, fdo21]
    fdw_vector = [fdw10, fdw20, fdw21]

    improvement_w = np.array(udw_vector) - np.array(fdw_vector)
    improvement_o = np.array(udo_vector) - np.array(fdo_vector)

    unfair_downgraded = np.array(udo_vector).sum()
    fair_downgraded = np.array(fdo_vector).sum()

    ans = {'unfair_downgraded': unfair_downgraded,
           'fair_downgraded': fair_downgraded,
           'unfair_1_matrix': unfair_1_matrix,
           'unfair_0_matrix': unfair_0_matrix,
           'fair_1_matrix': fair_1_matrix,
           'unfair_discriminated_losses': udo_vector,
           'unfair_priveleged_losses': udw_vector,
           'fair_discriminated_losses': fdo_vector,
           'fair_priveleged_losses': fdw_vector,
           'fair_0_matrix': fair_0_matrix,
           'improvement_1': improvement_w,
           'improvement_0': improvement_o,
           'unfair_one_accuracy': unfair_1_accuracy,
           'unfair_zero_accuracy': unfair_0_accuracy,
           'fair_one_accuracy': fair_1_accuracy,
           'fair_zero_accuracy': fair_0_accuracy,
           'unfair_discriminated_one_21': udw21,
           'unfair_discriminated_one_20': udw20,
           'unfair_discriminated_one_10': udw10,
           'unfair_upgraded_one_01': upw01,
           'unfair_upgraded_one_02': upw02,
           'unfair_upgraded_one_12': upw12,
           'unfair_discriminated_zero_21': udo21,
           'unfair_discriminated_zero_20': udo20,
           'unfair_discriminated_zero_10': udo10,
           'unfair_upgraded_zero_01': upo01,
           'unfair_upgraded_zero_02': upo02,
           'unfair_upgraded_zero_12': upo12,
           'fair_discriminated_one_21': fdw21,
           'fair_discriminated_one_20': fdw20,
           'fair_discriminated_one_10': fdw10,
           'fair_upgraded_one_01': fpw01,
           'fair_upgraded_one_02': fpw02,
           'fair_upgraded_one_12': fpw12,
           'fair_discriminated_zero_21': fdo21,
           'fair_discriminated_zero_20': fdo20,
           'fair_discriminated_zero_10': fdo10,
           'fair_upgraded_zero_01': fpo01,
           'fair_upgraded_zero_02': fpo02,
           'fair_upgraded_zero_12': fpo12}
    return ans
