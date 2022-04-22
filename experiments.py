"""Peform hyperparemeters search"""
# Started from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/search_hyperparams.py

import argparse
import os
import pandas as pd
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

# Run evaluation for best and worst models to get some differences
def eval_model(hp, hp_val):
    # Launch evaluation for hp_val
    model_dir = 'experiments/hyperparams/' + hp + '/' + hp + '_' + str(hp_val) + '/'
    cmd = "{python} evaluate.py --model_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)

def get_some_differences(hp, hp_max, hp_min):
    df_max = pd.read_csv('experiments/hyperparams/' + hp + '/' + hp + '_' + str(hp_max) + '/model_output.tsv', sep='\t')
    df_min = pd.read_csv('experiments/hyperparams/' + hp + '/' + hp + '_' + str(hp_min) + '/model_output.tsv', sep='\t')

    df_max.columns = ['word', 'gold', 'pred_max']
    df_min.columns = ['word', 'gold', 'pred_min']
    print('Some differences for %s - %s vs %s' % (hp, str(hp_min), str(hp_max)))
    df_diff = df_min.merge(df_max).dropna().drop_duplicates()
    print(df_diff[df_diff.pred_min != df_diff.pred_max][0:5])

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

        print('Training models...')
        for hyperpar in exp['param_list']:
            # Modify the relevant parameter in params
            setattr(params, exp_name, hyperpar)

            # Launch job (name has to be unique)
            job_name = (exp_name+"_{}").format(hyperpar)
            launch_training_job(exp['parent_dir'], args.data_dir, job_name, params, args.eval_metric)

            # Get output weighted average F1 (wa_f1) from metrics_val_best_weights.json
            wa_f1s.append(get_wa_f1(exp['parent_dir'] + job_name, args.eval_metric))

        print('Create plot...')
        utils.save_plot([str(hp) for hp in exp['param_list']], wa_f1s, exp['hp_name'], 'Weighted Average F1',
                        exp['plot_title'], exp_name+'.png', 'images/hyperparams/', 'bar')

        print('Evaluate best and worst hyperparams...')
        max_hp = exp['param_list'][wa_f1s.index(max(wa_f1s))]
        min_hp = exp['param_list'][wa_f1s.index(min(wa_f1s))]
        eval_model(exp_name, max_hp)
        eval_model(exp_name, min_hp)
        print('\n')
        get_some_differences(exp_name, max_hp, min_hp)