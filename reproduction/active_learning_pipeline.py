import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from acquisition_fn import move_indices
from reproduction.train_best_hyperparam import train_best_hyperparam, test_model


def avg_acc_std_AL(repetitions_al, acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations,
                   data_transfer_amount, seed, device, deterministic):
    accs = []
    for i in range(repetitions_al):
        curr_seed = seed + 10*i
        train_copy = copy.deepcopy(train_set)
        pool_copy = copy.deepcopy(pool_set)
        accs.append(record_accuracy_AL(acquisition_fn, train_copy, pool_copy, val_loader, test_loader, al_iterations, data_transfer_amount, curr_seed, device, deterministic))

    accs = np.array(accs)
    avg_accs = accs.mean(axis=0)
    std_accs = accs.std(axis=0)

    return avg_accs, std_accs


def record_accuracy_AL(acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations, data_transfer_amount, seed, device, deterministic=False):
    torch.manual_seed(seed)
    accuracies = []

    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    model = train_best_hyperparam(train_loader, val_loader, 20, seed, device, deterministic)

    accuracy = test_model(model, device, test_loader)
    accuracies.append(accuracy)
    print(f'Current length of train set: {len(train_set)}, current accuracy: {accuracy}')

    for i in range(al_iterations):
        # find best datapoints and move them to training data
        good_data_indices = acquisition_fn(pool_set, model, seed, device, data_transfer_amount)
        move_indices(train_set, pool_set, good_data_indices)

        # train new model with extended training data
        train_loader = DataLoader(train_set, batch_size=min(len(train_set), 128), shuffle=True)
        model = train_best_hyperparam(train_loader, val_loader, 20, seed, device, deterministic)

        accuracy = test_model(model, device, test_loader)
        accuracies.append(accuracy)
        print(f'Current length of train set: {len(train_set)}, current accuracy: {accuracy}')

    return accuracies
