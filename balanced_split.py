from torch.utils.data import Subset
import random
from collections import defaultdict

def make_random_balanced_split(dataset, train_per_class, val_size=100):
    class_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_to_indices[label].append(i)

    for cls in class_to_indices:
        random.shuffle(class_to_indices[cls])

    train_indices = []
    for cls, idxs in class_to_indices.items():
        train_indices.extend(idxs[:train_per_class])

    remaining = [i for i in range(len(dataset)) if i not in train_indices]
    random.shuffle(remaining)

    val_indices = remaining[:val_size]
    pool_indices = remaining[val_size:]

    train_sub = Subset(dataset, train_indices)
    val_sub   = Subset(dataset, val_indices)
    pool_sub  = Subset(dataset, pool_indices)

    return train_sub, val_sub, pool_sub
