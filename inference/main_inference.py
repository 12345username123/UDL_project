import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inference.active_learning_inference_pipeline import record_RSME_AL_inference
from inference.inference import analytic_inference_W, analytic_inference_predict, MFVI_W, MFVI_predict
from balanced_split import make_random_balanced_split


def main_inference():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    root = "./features"
    training_path = os.path.join(root, "processed", "training.pt")
    test_path     = os.path.join(root, "processed", "test.pt")

    train_feats, train_labels = torch.load(training_path)   # expect (N_train, 128), (N_train,)
    test_feats,  test_labels  = torch.load(test_path)

    train_feats = train_feats.float()
    test_feats  = test_feats.float()

    # Minimal wrapper dataset that returns label as a Python int
    class FeatureDatasetCPU(Dataset):
        def __init__(self, feats, labels):
            # keep as tensors (cpu) for fast indexing; convert labels to CPU if needed
            self.feats = feats if not feats.is_cuda else feats.cpu()
            self.labels = labels if not labels.is_cuda else labels.cpu()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x = self.feats[idx]
            y = int(self.labels[idx].item())   # <-- return a plain Python int
            return x, y


    train_dataset = FeatureDatasetCPU(train_feats, train_labels)
    test_dataset  = FeatureDatasetCPU(test_feats, test_labels)

    train_set, _, pool_set = make_random_balanced_split(train_dataset, train_per_class=2, val_size=0)

    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset))

    al_iterations = 100
    data_transfer_amount = 10

    print(f'Starting on calculating RSME and accuracy for analytic inference')
    RSMEs, accuracies = record_RSME_AL_inference(al_iterations, data_transfer_amount, analytic_inference_W, analytic_inference_predict, train_set,
                                     pool_set, test_loader, device, data_var=1.0, prior_var=1.0)
    np.save(f"./results/analytic_inference_RMSE.npy", RSMEs)

    train_dataset = FeatureDatasetCPU(train_feats, train_labels)
    test_dataset  = FeatureDatasetCPU(test_feats, test_labels)

    train_set, _, pool_set = make_random_balanced_split(train_dataset, train_per_class=2, val_size=0)

    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset))

    print(f'Starting on calculating RSME for MFVI')
    RSMEs, accuracies = record_RSME_AL_inference(al_iterations, data_transfer_amount, MFVI_W, MFVI_predict, train_set,
                                     pool_set, test_loader, device, data_var=1.0, prior_var=1.0)
    np.save(f"./results/MFVI_RMSE.npy", RSMEs)

