# Copyright 2024 Benjamin Leblanc

# This file is part of Binary Greedy Network (BGN).

# BGN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from utils import *
from warnings import simplefilter
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def launch(experiment_name, dataset_name, seed,n_folds, max_neur, max_hdlr, max_repl, max_coef_per_neur, patience):
    """
        The experiment launcher.

        Args:
            experiment_name (str): The name of the .txt file to write the results into.
            dataset_name (list of str): The dataset to train on. Choices: (from the UCI dataset repository):
                'bike_hour', 'carbon', 'diabete', 'housing', 'hung_pox', 'ist_stock_usd', 'parking', 'power_plant',
                'solar_flare', 'portfolio' or 'turbine'.
            seed (list of int): The random seed to use.
            n_folds (list of int): If 0: uses a 60-20-20 train-valid-test random split; if > 0: k-cross validation.
            max_neur (list of int): Maximum number of neurons in the layer.
            max_hdlr (list of int): Maximum number of hidden layers in the final predictor.
            max_repl (list of int): Maximum number of hyperplanes to be removed and replaced during a single iteration of
                                    the algorithm (see paper).
            max_coef_per_neur (list of int): Maximum number non-zero weights per hidden neuron; set to 0 to have all
                                                weights having non-zero values.
            patience (list of int): Patience parameter before ending the building of a layer.

        Returns:
            Tuple (train np.array,
                   validation np.array,
                   test np.array,
                   validation loss float,
                   number of non-zero params int,)
        """
    param_grid = ParameterGrid([{'dataset_name': dataset_name,
                                 'seed': seed,
                                 'n_folds': n_folds,
                                 'max_neur': max_neur,
                                 'max_hdlr': max_hdlr,
                                 'max_repl': max_repl,
                                 'max_coef_per_neur': max_coef_per_neur,
                                 'patience': patience}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset_name)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset_name']])
    n_tasks = len(param_grid)
    for i, task_dict in enumerate(param_grid):
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        cnt_nw = is_job_already_done(experiment_name, task_dict, fold_number=None)
        if cnt_nw >= max(task_dict['n_folds'], 1):
            print("Already done; passing...\n")
        else:
            train_load, test, dy, f_n = load_dataset(task_dict['dataset_name'], 0.80, task_dict['seed'],
                                                         task_dict['n_folds'])
            for fold_num in range(max(task_dict['n_folds'], 1)):
                if cnt_nw == 0:
                    if task_dict['n_folds'] > 0:
                        print(f"\tLaunching fold {fold_num + 1}/{task_dict['n_folds']}\n")
                    train, valid = train_valid_loaders(train_load[fold_num], train_split=0.75)
                    arch = ''
                    t_before = time()
                    tr_l, vd_l, te_l, nn = create_layer((train, valid, test[fold_num], f_n),
                                                       task_dict['dataset_name'],
                                                       task_dict['seed'],
                                                       task_dict['n_folds'],
                                                       fold_num,
                                                       task_dict['max_neur'],
                                                       arch,
                                                       0,
                                                       task_dict['max_repl'],
                                                       task_dict['max_coef_per_neur'],
                                                       task_dict['patience'],
                                                       dy,
                                                       experiment_name)
                    num_hid_lay = 1
                    file = open("results/" + str(experiment_name) + "_done.txt", "a")
                    file.write('BGN' + '\t' + task_dict['dataset_name'] + '\t' + str(task_dict['seed']) +
                               '\t' + str(task_dict['n_folds']) + '\t' + str(fold_num) +
                               '\t' + str(task_dict['max_neur']) + '\t' + str(num_hid_lay) +
                               '\t' + str(task_dict['max_repl']) + '\t' + str(task_dict['max_coef_per_neur']) + "\t" +
                               str(task_dict['patience']) + "\t" + str(tr_l) + "\t" + str(vd_l) + "\t" + str(te_l) +
                               "\t" + str(nn) + "\t" + str(time() - t_before) + "\n")
                    file.close()
launch(experiment_name='BGN_diabete',
       dataset_name=['diabete'],
       seed=[20241202],
       n_folds=[0],
       max_neur=[15],
       max_hdlr=[1],
       max_repl=[100],
       max_coef_per_neur=[2],
       patience=[15])
