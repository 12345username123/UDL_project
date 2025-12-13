import os

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Feature_Extractor

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST("../data", train=True, transform=transform, download=True)
test_data = datasets.MNIST("../data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = Feature_Extractor(in_channels = 1, num_classes = 10).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
model.train()

for i in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'Training epoch {i+1} of feature extractor completed')


model.eval()
total = 0
correct = 0
for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model.softmax(model(images))
        predictions = predictions.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f'Accuracy of feature extractor on training data: {correct * 100.0 / total}')

torch.save(model.state_dict(), "../model/feature_extractor.pth")

all_feats = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        feats = model.features(images)
        all_feats.append(feats.detach().cpu())
        all_labels.append(labels.detach().cpu())

feats_tensor = torch.cat(all_feats, dim=0)
labels_tensor = torch.cat(all_labels, dim=0)

FEATURES_ROOT = "../features"
processed_dir = os.path.join(FEATURES_ROOT, "processed")
os.makedirs(processed_dir, exist_ok=True)
out_fname = "training.pt"
out_path = os.path.join(processed_dir, out_fname)
torch.save((feats_tensor, labels_tensor), out_path)
print(f"Saved train features -> {out_path} (feats shape {feats_tensor.shape}, labels shape {labels_tensor.shape})")


all_feats_test = []
all_labels_test = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        feats = model.features(images)
        all_feats_test.append(feats.detach().cpu())
        all_labels_test.append(labels.detach().cpu())

feats_tensor_test = torch.cat(all_feats_test, dim=0)
labels_tensor_test = torch.cat(all_labels_test, dim=0)

FEATURES_ROOT = "../features"
processed_dir = os.path.join(FEATURES_ROOT, "processed")
os.makedirs(processed_dir, exist_ok=True)
out_fname = "test.pt"
out_path = os.path.join(processed_dir, out_fname)
torch.save((feats_tensor_test, labels_tensor_test), out_path)
print(f"Saved test features -> {out_path} (feats shape {feats_tensor_test.shape}, labels shape {labels_tensor_test.shape})")