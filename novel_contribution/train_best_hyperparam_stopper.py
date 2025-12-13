import numpy as np
import torch
from torch import nn

from model import CNN_Bayesian_Stopper


def train_best_hyperparam_stopper(train_loader, val_loader, num_epochs, seed, device):
    criterion = nn.CrossEntropyLoss()

    weight_decays = [1e-5, 1e-4, 1e-3]
    val_acc = []
    models = []

    for weight_decay in weight_decays:
        torch.manual_seed(seed)
        model = CNN_Bayesian_Stopper().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        val_acc.append(test_model(model, device, val_loader))
        models.append(model)

    return models[np.argmax(val_acc)]


def test_model(model, device, test_loader):
    total = 0
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = (model.predict_proba_test_stopper(images)).mean(dim=0)
        predictions = predictions.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total



