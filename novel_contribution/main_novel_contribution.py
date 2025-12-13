import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from acquisition_fn import mean_std, bald, max_entropy, variation_ratio, bald_stopper
from novel_contribution.active_learning_pipeline_mciter_count import avg_acc_mciters_AL
from balanced_split import make_random_balanced_split


def main_novel_contribution():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST("./data", train=True, transform=transform, download=True)
    num_samples = len(train_data) // 10
    indices = torch.randperm(len(train_data))[:num_samples]
    train_data = Subset(train_data, indices)

    test_data = datasets.MNIST("./data", train=False, transform=transform, download=True)
    num_samples = len(test_data) // 10
    indices = torch.randperm(len(test_data))[:num_samples]
    test_data = Subset(test_data, indices)

    train_set, val_set, pool_set = make_random_balanced_split(train_data, train_per_class=2, val_size=100)
    val_loader = DataLoader(val_set, batch_size=100)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    al_iterations = 100
    data_transfer_amount = 10
    repetitions_al = 3

    acquisition_fns = {
        'bald_stopper': bald_stopper,
    }

    for acquisition_fn_name, acquisition_fn in acquisition_fns.items():
        print(f'Starting on determining accuracy for {acquisition_fn_name}')
        acc, mc_iters = avg_acc_mciters_AL(repetitions_al, acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations,
                                      data_transfer_amount, seed, device)
        print(f'Average accuracies and std for {acquisition_fn_name}:')
        print(f'{acc}')
        print(f'{mc_iters}')
        np.save(f"results/{acquisition_fn_name}_acc.npy", acc)
        np.save(f"results/{acquisition_fn_name}_mciters.npy", mc_iters)