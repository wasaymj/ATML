import numpy as np
def get_train_val_indices(targets, seed=6304, num_classes=10):
    train_indices = []
    val_indices = []
    g = np.random.RandomState(seed)
    for c in range(num_classes):
        c_idx = np.where(targets == c)[0]
        g.shuffle(c_idx)
        split = int(0.9 * len(c_idx))
        train_indices.extend(c_idx[:split])
        val_indices.extend(c_idx[split:])
    return train_indices, val_indices
