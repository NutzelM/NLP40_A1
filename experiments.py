"""Peform hyperparemeters search"""
# Started from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/search_hyperparams.py

import argparse
import os
from subprocess import check_call
import sys
import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    experiments = {
        'learning_rate': {
            'parent_dir': 'experiments/hyperparams/learning_rate/',
            'param_list': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1],
            'hp_name': 'Learning Rates',
            'plot_title': 'Weighted average F1 score for different learning rates'
        },
        'embedding_dim': {
            'parent_dir': 'experiments/hyperparams/embedding_dim/',
            'param_list': range(10, 90, 10),
            'hp_name': 'Embedding dimensions',
            'plot_title': 'Weighted average F1 score for different embedding dimensions'
        }
    }

    for exp_name, exp in experiments.items():
        print('---------------Start %s experiment---------------' % exp_name)
        json_path = os.path.join(exp['parent_dir'], 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)

        # Perform hyperparameter search
        wa_f1s = []

        for hyperpar in exp['param_list']:
            # Modify the relevant parameter in params
            setattr(params, exp_name, hyperpar)

            # Launch job (name has to be unique)
            job_name = (exp_name+"_{}").format(hyperpar)
            launch_training_job(exp['parent_dir'], args.data_dir, job_name, params, args.eval_metric)

            # Get output weighted average F1 (wa_f1) from metrics_val_best_weights.json
            wa_f1s.append(get_wa_f1(exp['parent_dir'] + job_name, args.eval_metric))

        utils.save_plot([str(hp) for hp in exp['param_list']], wa_f1s, exp['hp_name'], 'Weighted Average F1',
                        exp['plot_title'], exp_name+'.png', 'images/hyperparams/', 'bar')