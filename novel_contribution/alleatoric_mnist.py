import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset

from acquisition_fn import mean_std, bald, max_entropy, variation_ratio, bald_stopper, max_entropy_avg_iters
from novel_contribution.active_learning_pipeline_mciter_count import avg_acc_mciters_AL
from balanced_split import make_random_balanced_split
from reproduction.active_learning_pipeline import avg_acc_std_AL


def main_novel_contribution_alleatoric_data():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # modify the dataset:
    def build_modified(dataset):
        n = len(dataset) // 10
        idx = torch.randperm(len(dataset))[:n]

        imgs = []
        labs = []

        for i in idx.tolist():
            img, lbl = dataset[i]
            lbl = int(lbl)
            if lbl in (1, 2, 3, 4, 5, 6):
                continue
            if lbl == 0:
                imgs.append(img); labs.append(0)
                for new_lbl in (1, 2, 3, 4, 5, 6):
                    imgs.append(img); labs.append(new_lbl)
            else:
                imgs.append(img); labs.append(lbl)

        perm = torch.randperm(len(labs))
        imgs = [imgs[i] for i in perm.tolist()]
        labs = torch.tensor([labs[i] for i in perm.tolist()], dtype=torch.long)

        class SimpleListDataset(Dataset):
            def __init__(self, imgs, labs):
                self.imgs = imgs; self.labs = labs
            def __len__(self): return len(self.labs)
            def __getitem__(self, i): return self.imgs[i], int(self.labs[i])

        return SimpleListDataset(imgs, labs)

    full_train = datasets.MNIST("./data", train=True, transform=transform, download=True)
    full_test  = datasets.MNIST("./data", train=False,transform=transform, download=True)

    train_data = build_modified(full_train)
    test_data  = build_modified(full_test)

    train_set, val_set, pool_set = make_random_balanced_split(train_data, train_per_class=2, val_size=100)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    al_iterations = 50
    data_transfer_amount = 10
    repetitions_al = 3


    acquisition_fns = {
        'bald': bald,
        'max_entropy': max_entropy,
    }

    for acquisition_fn_name, acquisition_fn in acquisition_fns.items():
        print(f'Starting on determining accuracy for {acquisition_fn_name}')
        acc, _ = avg_acc_std_AL(repetitions_al, acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations,
                                  data_transfer_amount, seed, device, deterministic=False)
        print(f'Average accuracies for {acquisition_fn_name}:')
        print(f'{acc}')
        np.save(f"results/{acquisition_fn_name}_aleatoric_acc.npy", acc)


    train_set, val_set, pool_set = make_random_balanced_split(train_data, train_per_class=2, val_size=100)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    acquisition_fns = {
        'max_entropy_avg_mciters': max_entropy_avg_iters,
    }

    for acquisition_fn_name, acquisition_fn in acquisition_fns.items():
        print(f'Starting on determining accuracy for {acquisition_fn_name}')
        acc, mc_iters = avg_acc_mciters_AL(repetitions_al, acquisition_fn, train_set, pool_set, val_loader, test_loader, al_iterations,
                                           data_transfer_amount, seed, device)
        print(f'Average accuracies for {acquisition_fn_name}:')
        print(f'{acc}')
        np.save(f"results/{acquisition_fn_name}_alleatoric_acc.npy", acc)