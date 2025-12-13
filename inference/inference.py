import torch


def analytic_inference_W(features, labels, data_var, prior_var, device):
    # features have dim=(Batch, 128), labels have dim=(Batch, 10) --> W_var_block has dim=(128,128)
    W_var_block = torch.linalg.inv(features.T @ features / data_var + torch.eye(features.shape[1]).to(device) / prior_var)
    # apply same variance block to every label simultaneously
    W_mean = W_var_block @ features.T @ labels / data_var
    # return mean (dim=(128,10)) and covariance matrix block (dim=(128,128))
    return W_mean, W_var_block


def analytic_inference_predict(data, W_mean, W_var_block, data_var):
    # data has dim=(Batch, 128)
    prediction_mean = data @ W_mean
    # use einsum to calculate scalar variance for each datapoint, since full covariance matrix is
    # diagonal with this entry everywhere
    prediction_var = data_var + torch.einsum("bf,fg,bg->b", data, W_var_block, data)
    return prediction_mean, prediction_var


def MFVI_W(features, labels, data_var, prior_var, device):
    # features have dim=(Batch, 128), labels have dim=(Batch, 10)
    b = features.T @ labels
    A = features.T @ features + torch.eye(features.shape[1]).to(device) * data_var / prior_var
    M = torch.linalg.solve(A, b)

    # calculate sigma_kd for each k
    # sum features over the batch --> dim=(128,)
    a = torch.sum(features**2, dim=0)
    prec_k = (1.0 / prior_var) + (a / data_var)
    S_k = 1.0 / prec_k
    S_k = S_k.clamp(min=0.0)

    S_diag = torch.diag(S_k)

    return M, S_diag

def MFVI_predict(data, M, S, data_var):
    # data has dim=(Batch, 128)
    prediction_mean = data @ M
    prediction_var = data_var + torch.einsum("bf,fg,bg->b", data, S, data)
    return prediction_mean, prediction_var