import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        flattened_size = 32 * 11 * 11
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        # to train with cross entropy loss, I apply the best practice and train on the logits
        return logits

    def predict_proba(self, x, mc_iters=20):
        pass


class CNN_Bayesian(CNN):
    # to predict and do MC dropout, I use the probabilities obtained by soft maxing
    # to obtain real prediction, use mean on result
    # Important: self.train to keep dropout active at evaluation time
    def predict_proba(self, x, mc_iters=20):
        self.train()
        with torch.no_grad():
            probs = [self.softmax(self(x)) for _ in range(mc_iters)]
            probs = torch.stack(probs)
        # shape: (mc_iters, batch, class)
        return probs


class CNN_Deterministic(CNN):
    # to predict with a deterministic model, put in eval mode (deactivates mc) and don't do MC
    # to model formally the deterministic model parameters being a dirac function and to use the same
    # acquisition functions: add additional dimension --> mean over it will return the 'true' probabilities
    def predict_proba(self, x, mc_iters=20):
        self.eval()
        with torch.no_grad():
            probs = [self.softmax(self(x))]
            probs = torch.stack(probs)
        # shape: (1, batch, class)
        return probs


class Feature_Extractor(CNN):
    def features(self, x):
        self.eval()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        features = self.relu(self.fc1(x))
        return features


class CNN_Bayesian_Stopper(CNN):
    # Assumes that x has a batch size of 1
    def predict_proba(self, x, mc_iters=20, epsilon=1e-3):
        self.train()
        probs_list = []

        with torch.no_grad():
            for t in range(mc_iters):
                p = self.softmax(self(x))  # (batch, n_classes)
                probs_list.append(p)

                if t > 0:
                    prev = torch.stack(probs_list[:-1])   # (t, batch, n_classes)
                    diffs = (prev - p.unsqueeze(0)).pow(2) # (t, batch, n_classes)

                    # mean MSE across previous draws, samples and classes (scalar)
                    mse_mean = diffs.mean().item()

                    if mse_mean < epsilon:
                        break

        probs = torch.stack(probs_list)  # (n_draws, batch, n_classes)
        return probs

    # when testing, do all MC iterations
    def predict_proba_test_stopper(self, x, mc_iters=20):
        self.train()
        with torch.no_grad():
            probs = [self.softmax(self(x)) for _ in range(mc_iters)]
            probs = torch.stack(probs)
        # shape: (mc_iters, batch, class)
        return probs