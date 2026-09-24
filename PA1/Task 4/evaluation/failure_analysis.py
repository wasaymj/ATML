import json
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def save_failures(data, scores, threshold, group_name, out_dir, dataset):
    accepted_mask = (scores <= threshold)
    accepted_indices = accepted_mask.nonzero(as_tuple=True)[0]
    failures = []
    failed_imgs = []
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    for idx in accepted_indices[:3]:
        pred_class_id = data['logits'][idx, :10].argmax().item()
        true_class_id = data['labels'][idx].item()
        score = scores[idx].item()
        img_tensor = dataset[idx.item()][0]
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        failed_imgs.append(img)
        predicted_known_class = CIFAR10_CLASSES[pred_class_id]
        unknown_class = (CIFAR100_CLASSES[true_class_id] if true_class_id < len(CIFAR100_CLASSES) else f'cifar100_class_{true_class_id}')
        failures.append({'unknown_class': unknown_class, 'predicted_known_class': predicted_known_class, 'score': score, 'threshold': threshold})

    path_json = os.path.join(out_dir, f'failures_vanilla_mls_{group_name}.json')
    with open(path_json, 'w') as f:
        json.dump(failures, f, indent=2)

    if len(failed_imgs) > 0:
        fig, axes = plt.subplots(1, len(failed_imgs), figsize=(3.5 * len(failed_imgs), 3.5))
        if len(failed_imgs) == 1: axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(failed_imgs[i].permute(1, 2, 0).numpy())
            f = failures[i]
            title = f"True: {f['unknown_class']}
Pred: {f['predicted_known_class']}
Score: {f['score']:.2f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        plt.tight_layout()
        path_img = os.path.join(out_dir, f'failures_vanilla_mls_{group_name}.png')
        plt.savefig(path_img, dpi=150)
        plt.close()
    print(f'  [failures] Saved {len(failures)} {group_name} failures (JSON & PNG)')
    sys.stdout.flush()
