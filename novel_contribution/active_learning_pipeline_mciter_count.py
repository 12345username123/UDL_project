import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from acquisition_fn import move_indices
from novel_contribution.train_best_hyperparam_stopper import train_best_hyperparam_stopper, test_model


def avg_acc_mciters_AL(repetitions_al, acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations,
                       data_transfer_amount, seed, device):
    accs = []
    mc_iters_poolset = []
    for i in range(repetitions_al):
        curr_seed = seed + 10*i
        train_copy = copy.deepcopy(train_set)
        pool_copy = copy.deepcopy(pool_set)
        acc, mc_iter_poolset = record_accuracy_mciters_AL(acquisition_fn, train_copy, pool_copy, val_loader, test_loader, al_iterations, data_transfer_amount, curr_seed, device)
        accs.append(acc)
        mc_iters_poolset.append(mc_iter_poolset)

    accs = np.array(accs)
    avg_accs = accs.mean(axis=0)
    mc_iters_poolset = np.array(mc_iters_poolset)
    avg_mc_iters_poolset = mc_iters_poolset.mean(axis=0)

    return avg_accs, avg_mc_iters_poolset


def record_accuracy_mciters_AL(acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations, data_transfer_amount, seed, device):
    torch.manual_seed(seed)
    accuracies = []
    avg_mc_iters = []

    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    model = train_best_hyperparam_stopper(train_loader, val_loader, 20, seed, device)

    accuracy = test_model(model, device, test_loader)
    accuracies.append(accuracy)
    print(f'Current length of train set: {len(train_set)}, current accuracy: {accuracy}')

    for i in range(al_iterations):
        # find best datapoints and move them to training data
        good_data_indices, avg_mc_iter = acquisition_fn(pool_set, model, seed, device, data_transfer_amount)
        avg_mc_iters.append(avg_mc_iter)
        move_indices(train_set, pool_set, good_data_indices)

        # train new model with extended training data
        train_loader = DataLoader(train_set, batch_size=min(len(train_set), 128), shuffle=True)
        model = train_best_hyperparam_stopper(train_loader, val_loader, 20, seed, device)

        accuracy = test_model(model, device, test_loader)
        accuracies.append(accuracy)
        print(f'Current length of train set: {len(train_set)}, current accuracy: {accuracy}, avg mc iters: {avg_mc_iter}')

    return accuracies, avg_mc_iters
