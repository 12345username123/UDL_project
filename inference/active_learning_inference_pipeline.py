import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from acquisition_fn import predictive_variance_acquisition, move_indices


def record_RSME_AL_inference(al_iterations, data_transfer_amount, inference_W_fn, inference_predict_fn,train_set,
                             pool_set, test_loader, device, data_var=1.0, prior_var=1.0):
    RSMEs = []
    train_copy = copy.deepcopy(train_set)
    pool_copy = copy.deepcopy(pool_set)

    train_loader = DataLoader(train_copy, batch_size=len(train_set))
    features, labels = next(iter(train_loader))
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
    features, labels = features.to(device), labels_one_hot.to(device)

    W_mean, W_var_block = inference_W_fn(features, labels, data_var, prior_var, device)
    criterion = nn.MSELoss(reduction='sum')
    mse_sum = 0.0
    n_samples = 0
    for features, labels in test_loader:
        features, labels = features.to(device), torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
        predictions, _ = inference_predict_fn(features, W_mean, W_var_block, data_var)
        mse_sum += criterion(predictions, labels).item()
        n_samples += labels.size(0)
    RSMEs.append((mse_sum / n_samples)**0.5)

    print(f'Current length of train set: {len(train_set)}, current RMSE: {RSMEs[-1]}')

    for i in range(al_iterations):
        # find best datapoints and move them to training data
        pool_loader = DataLoader(pool_copy, batch_size=len(pool_copy))
        pool_set_features, _ = next(iter(pool_loader))
        pool_set_features = pool_set_features.to(device)
        good_data_indices = predictive_variance_acquisition(pool_set_features, inference_predict_fn, W_mean,
                                                            W_var_block, data_var, data_transfer_amount)
        move_indices(train_set, pool_set, good_data_indices)

        train_loader = DataLoader(train_set, batch_size=len(train_set))
        features, labels = next(iter(train_loader))
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10)
        features, labels = features.to(device), labels_one_hot.to(device).float()

        W_mean, W_var_block = inference_W_fn(features, labels, data_var, prior_var, device)
        criterion = nn.MSELoss(reduction='sum')
        mse_sum = 0.0
        n_samples = 0
        for features, labels in test_loader:
            features, labels = features.to(device), torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            predictions, _ = inference_predict_fn(features, W_mean, W_var_block, data_var)
            mse_sum += criterion(predictions, labels).item()
            n_samples += labels.size(0)
        RSMEs.append((mse_sum / n_samples)**0.5)

        print(f'Current length of train set: {len(train_set)}, current RMSE: {RSMEs[-1]}')

    return RSMEs
