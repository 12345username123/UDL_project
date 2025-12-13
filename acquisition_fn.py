import torch
from torch.utils.data import DataLoader


def move_indices(train_set, pool_set, indices):
    picked_dataset_indices = [pool_set.indices[i] for i in indices]
    train_set.indices.extend(picked_dataset_indices)
    pool_set.indices = [
        idx for j, idx in enumerate(pool_set.indices)
        if j not in indices
    ]


def top_k_random_return(scores, data_transfer_amount):
    topk_vals, _ = torch.topk(scores, data_transfer_amount)
    threshold = topk_vals[-1].item()
    eligible_indices = torch.nonzero(scores >= threshold, as_tuple=False).squeeze(1)

    chosen_indices = eligible_indices[torch.randperm(len(eligible_indices))[:data_transfer_amount]]

    chosen_indices = chosen_indices.tolist()
    return chosen_indices



def max_entropy(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=batch_size, shuffle=False)
    scores = []

    eps = 1e-12
    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        # clamp to not get too small values in log
        probabilities = model.predict_proba(x, mc_iters=mc_iters).clamp(min=eps)

        # calculate entropy over average predictions
        # shape: (batch, class)
        p_mean = probabilities.mean(dim=0)
        # iterate over classes --> dim = 1, leaves shape: (batch, )
        H_mean = - (p_mean * p_mean.log()).sum(dim=1)

        scores.append(H_mean)

    scores = torch.cat(scores, dim=0)

    return top_k_random_return(scores, data_transfer_amount)


def max_entropy_avg_iters(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=1, shuffle=False)
    scores = []
    num_mc_iters = 0
    num_data = 0

    eps = 1e-12
    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        # clamp to not get too small values in log
        probabilities = model.predict_proba(x, mc_iters=mc_iters).clamp(min=eps)
        num_mc_iters += probabilities.shape[0]
        num_data += 1

        # calculate entropy over average predictions
        # shape: (batch, class)
        p_mean = probabilities.mean(dim=0)
        # iterate over classes --> dim = 1, leaves shape: (batch, )
        H_mean = - (p_mean * p_mean.log()).sum(dim=1)

        H_mean = H_mean * (probabilities.shape[0] / 20)**2

        scores.append(H_mean)

    scores = torch.cat(scores, dim=0)
    avg_mc_iters = num_mc_iters / num_data

    return top_k_random_return(scores, data_transfer_amount), avg_mc_iters


def bald(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=batch_size, shuffle=False)
    scores = []

    eps = 1e-12
    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        # clamp to not get too small values in log
        probabilities = model.predict_proba(x, mc_iters=mc_iters).clamp(min=eps)

        # calculate entropy over average predictions
        # shape: (batch, class)
        p_mean = probabilities.mean(dim=0)
        # iterate over classes --> dim = 1, leaves shape: (batch, )
        H_mean = - (p_mean * p_mean.log()).sum(dim=1)

        # calculate entropy over one set of predictions --> iterate over classes, dim=2
        # shape: (mc_iters, batch)
        H_t = - (probabilities * probabilities.log()).sum(dim=2)
        # expected entropy is average over mc_iters --> iterate over dim=0, leaves shape: (batch, )
        H_expected = H_t.mean(dim=0)

        bald_batch = H_mean - H_expected
        scores.append(bald_batch)

    scores = torch.cat(scores, dim=0)

    return top_k_random_return(scores, data_transfer_amount)


def bald_stopper(pool_set, model, seed, device, data_transfer_amount, mc_iters=20):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=1, shuffle=False)
    scores = []
    num_mc_iters = 0
    num_data = 0

    eps = 1e-12
    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        # clamp to not get too small values in log
        probabilities = model.predict_proba(x, mc_iters=mc_iters).clamp(min=eps)
        num_mc_iters += probabilities.shape[0]
        num_data += 1

        # calculate entropy over average predictions
        # shape: (batch, class)
        p_mean = probabilities.mean(dim=0)
        # iterate over classes --> dim = 1, leaves shape: (batch, )
        H_mean = - (p_mean * p_mean.log()).sum(dim=1)

        # calculate entropy over one set of predictions --> iterate over classes, dim=2
        # shape: (mc_iters, batch)
        H_t = - (probabilities * probabilities.log()).sum(dim=2)
        # expected entropy is average over mc_iters --> iterate over dim=0, leaves shape: (batch, )
        H_expected = H_t.mean(dim=0)

        bald_batch = H_mean - H_expected
        scores.append(bald_batch)

    scores = torch.cat(scores, dim=0)
    avg_mc_iters = num_mc_iters / num_data

    return top_k_random_return(scores, data_transfer_amount), avg_mc_iters


def variation_ratio(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=batch_size, shuffle=False)
    scores = []

    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        probabilities = model.predict_proba(x, mc_iters=mc_iters)

        # calculate entropy over average predictions
        # shape: (batch, class)
        p_mean = probabilities.mean(dim=0)

        # take probability of maximum class --> dim=1
        max_prob = p_mean.max(dim=1).values

        scores.append(1 - max_prob)

    scores = torch.cat(scores, dim=0)

    return top_k_random_return(scores, data_transfer_amount)


def mean_std(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)

    loader = DataLoader(pool_set, batch_size=batch_size, shuffle=False)
    scores = []

    for xb, _ in loader:
        x = xb.to(device)
        # shape: (mc_iters, batch, class)
        probabilities = model.predict_proba(x, mc_iters=mc_iters)

        # calculate expected values for each class --> iterate over dim=0, leaves shape: (batch, class)
        mean_p = probabilities.mean(dim=0)
        mean_p2 = (probabilities * probabilities).mean(dim=0)
        # Variance per class: sqrt(E[p^2] - (E[p])^2)
        var_class = mean_p2 - mean_p * mean_p
        # clamp to avoid numerical instability in sqrt
        var_class = var_class.clamp(min=0.0)
        std_class = var_class.sqrt()
        # average over classes
        sample_scores = std_class.mean(dim=1)
        scores.append(sample_scores)

    scores = torch.cat(scores, dim=0)

    return top_k_random_return(scores, data_transfer_amount)


def random_acquisition(pool_set, model, seed, device, data_transfer_amount, mc_iters=20, batch_size=128):
    torch.manual_seed(seed)
    samples = torch.randperm(len(pool_set))
    return samples[:data_transfer_amount].tolist()


def predictive_variance_acquisition(pool_set_features, inference_predict_fn, W_mean, W_var_block, data_var, data_transfer_amount):
    _, pred_var = inference_predict_fn(pool_set_features, W_mean, W_var_block, data_var)

    return top_k_random_return(pred_var, data_transfer_amount)