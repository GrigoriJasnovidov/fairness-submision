import random
from lightgbm import LGBMClassifier

from fmclp.main import fmclp
from fmclp.write_results import write_res, general_results_write


def run_experiment(dataset,
                   number_experiments,
                   multiplier,
                   number_iterations,
                   interior_classifier,
                   folder=None,
                   dataset_name=None,
                   initial_classifier=LGBMClassifier()):
    results = []

    for i in range(number_experiments):
        main_state = random.choice(range(1000))
        res = fmclp(dataset=dataset,
                    estimator=initial_classifier,
                    number_iterations=number_iterations,
                    prefit=False,
                    interior_classifier=interior_classifier,
                    verbose=False,
                    multiplier=multiplier,
                    random_state=main_state)
        results.append(res)
        name = f"{folder}/{dataset_name}_№'{i + 1}"
        write_res(res=res,
                  name=name,
                  main_state=main_state,
                  multiplier=multiplier,
                  interior_classifier=interior_classifier)
        print(i + 1)
    general_results_write(name=f"{folder}/{dataset_name}",
                          dataset_name=dataset_name,
                          classifier='lgb',
                          number_iterations=number_experiments,
                          multiplier=multiplier,
                          interior_classifier=interior_classifier,
                          results=results)
    return results
