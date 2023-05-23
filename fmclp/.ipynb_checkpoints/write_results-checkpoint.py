import numpy as np

from fmclp.benefit import benefit


def write_res(res, name, main_state, multiplier, interior_classifier):
    b = benefit(res)
    file = open(f"{name}.txt", 'w')
    file.write(f"""unfair_diff: {res['fairness_of_initial_classifier']['diff']}
unfair_ratio: {res['fairness_of_initial_classifier']['ratio']}
unfair_var: {res['fairness_of_initial_classifier']['variation']}
unfair_accuracy: {res['accuracy_of_initial_classifier']}
fair_diff: {res['fairness_of_fair_classifier']['diff']}
fair_ratio: {res['fairness_of_fair_classifier']['ratio']}
fair_var: {res['fairness_of_fair_classifier']['variation']}
fair_accuracy: {res['accuracy_of_fair_classifier']}
interior_classifier: {interior_classifier}
multiplier: {multiplier}
main_state: {main_state}
unfair_discriminated_group_losses: {b['unfair_discriminated_losses']}
fair_discriminated_group_losses: {b['fair_discriminated_losses']}
discrimanted_group_losses_improvement: {b['improvement_0']}
unfair_discriminated_downgraded: {b['unfair_downgraded']}
fair_discriminated_downgraded: {b['fair_downgraded']}
    """)
    file.close()
    res['fairness_of_fair_classifier']['df'].to_csv(f"{name} cuae-metric-fair.csv")
    res['fairness_of_initial_classifier']['df'].to_csv(f"{name} cuae-metric-unfair.csv")


def general_results_write(name, dataset_name, classifier, number_iterations, multiplier,
                          interior_classifier, results):
    fair_acc = []
    unfair_acc = []
    fair_var = []
    unfair_var = []
    fair_diff = []
    unfair_diff = []
    fair_ratio = []
    unfair_ratio = []
    improvement = []
    unfair_downgraded = []
    fair_downgraded = []
    for x in results:
        fair_acc.append(x['accuracy_of_fair_classifier'])
        unfair_acc.append(x['accuracy_of_initial_classifier'])
        fair_var.append(x['fairness_of_fair_classifier']['variation'])
        unfair_var.append(x['fairness_of_initial_classifier']['variation'])
        fair_diff.append(x['fairness_of_fair_classifier']['diff'])
        unfair_diff.append(x['fairness_of_initial_classifier']['diff'])
        fair_ratio.append(x['fairness_of_fair_classifier']['ratio'])
        unfair_ratio.append(x['fairness_of_initial_classifier']['ratio'])
        improvement.append(benefit(x)['improvement_0'].sum())
        unfair_downgraded.append(benefit(x)['unfair_downgraded'])
        fair_downgraded.append(benefit(x)['fair_downgraded'])
    fair_acc = np.array(fair_acc)
    unfair_acc = np.array(unfair_acc)
    fair_var = np.array(fair_var)
    unfair_var = np.array(unfair_var)
    fair_diff = np.array(fair_diff)
    unfair_diff = np.array(unfair_diff)
    fair_ratio = np.array(fair_ratio)
    unfair_ratio = np.array(unfair_ratio)
    improvement = np.array(improvement)
    unfair_downgraded = np.array(unfair_downgraded)
    fair_downgraded = np.array(fair_downgraded)
    file = open(name, 'w')
    file.write(
        f"""dataset for initial classifier training: {dataset_name} 
classifier: {classifier}
number_iterations: {number_iterations}
multiplier: {multiplier}
interior_classifier: {interior_classifier}
number_experiments: {len(results)}

fair_accuracy_mean: {fair_acc.mean()}
fair_diff_mean: {fair_diff.mean()}
fair_ratio_mean: {fair_ratio.mean()}
fair_var_mean: {fair_var.mean()}
fair_accuracy_std: {fair_acc.std()}
fair_diff_std: {fair_diff.std()}
fair_ratio_std: {fair_ratio.std()}
fair_var_std: {fair_var.std()}

unfair_accuracy_mean: {unfair_acc.mean()}
unfair_diff_mean: {unfair_diff.mean()}
unfair_ratio_mean: {unfair_ratio.mean()}
unfair_var_mean: {unfair_var.mean()}
unfair_accuracy_std: {unfair_acc.std()}
unfair_diff_std: {unfair_diff.std()}
unfair_ratio_std: {unfair_ratio.std()}
unfair_var_std: {unfair_var.std()}

(fair_accuracy-unfair_accuracy)_mean: {(fair_acc - unfair_acc).mean()}
(fair_accuracy-unfair_accuracy)_std: {(fair_acc - unfair_acc).std()}
(fair_diff-unfair_diff)_mean: {(fair_diff - unfair_diff).mean()}
(fair_diff-unfair_diff)_std: {(fair_diff - unfair_diff).std()}
(fair_ratio-unfair_ratio)_mean: {(fair_ratio - unfair_ratio).mean()}
(fair_ratio-unfair_ratio)_std: {(fair_ratio - unfair_ratio).std()}
(fair_variation-unfair_variation)_mean: {(fair_var - unfair_var).mean()}
(fair_varitaion-unfair_variation)_std: {(fair_var - unfair_var).std()}

fair_diff: {fair_diff}
fair_ratio: {fair_ratio}
fair_variation: {fair_var}
fair_accuracy: {fair_acc}
unfair_diff: {unfair_diff}
unfair_ratio: {unfair_ratio}
unfair_variation: {unfair_var}
unfair_accuracy: {unfair_acc}

discrimanted_group_losses_improvement: {improvement}
discrimanted_group_losses_improvement_mean: {improvement.mean()}
discrimanted_group_losses_improvement_std: {improvement.std()}

unfair_downgraded: {unfair_downgraded}
fair_downgraded: {fair_downgraded}
""")
    file.close()
