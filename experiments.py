"""Peform hyperparemeters search"""
# Started from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/search_hyperparams.py

import argparse
import os
from subprocess import check_call
import sys
import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir_lr', default='experiments/hyperparams/learning_rate/',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/preprocessed/', help="Directory containing the dataset")
parser.add_argument('--eval_metric', default='wa_f1', help="Evaluation metric (accuracy/wa_f1)")


def launch_training_job(parent_dir, data_dir, job_name, params, eval_metric):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --eval_metric {eval_metric}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, eval_metric=eval_metric)
    print(cmd)
    check_call(cmd, shell=True)

# Get weighted average F1 score from metrics JSON for experiment in dir_exp
def get_wa_f1(dir_exp, eval_metric):
    json_metrics_path = os.path.join(dir_exp, 'metrics_val_best_weights.json')
    assert os.path.isfile(json_metrics_path), "No json configuration file found at {}".format(json_metrics_path)
    metrics = utils.Params(json_metrics_path)
    return round(getattr(metrics, eval_metric, None),2)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir_lr, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over learning rate
    learning_rates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    wa_f1s = []

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir_lr, args.data_dir, job_name, params, args.eval_metric)

        # Get output weighted average F1 (wa_f1) from metrics_val_best_weights.json
        wa_f1s.append(get_wa_f1(args.parent_dir_lr + job_name, args.eval_metric))

    utils.save_plot([str(lr) for lr in learning_rates], wa_f1s, 'Learning Rates', 'Weighted Average F1',
                 'Weighted average F1 score for different learning rates', 'learning_rates.png', 'images/hyperparams/', 'bar')