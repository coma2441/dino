import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits

from sklearn.linear_model import LogisticRegression
from lora import LoRA_ViT_timm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='./labelled crops 20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--destination', default='', type=str, help='Destination folder to save results')
    parser.add_argument('--num_bins', default=25, type=int, help='Number of bins for calibration plot')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    if args.lora_rank is not None:
        model = LoRA_ViT_timm(model, r=args.lora_rank)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    
    # ============ extract features ... ============
    print("Extracting features for train set...")

    train_features = []
    train_labels = []

    for samples, labels in data_loader_train:
        train_features.append(model(samples).detach().numpy())
        train_labels.append(labels.detach().numpy())
        
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_filenames = [os.path.basename(f[0]) for f in dataset_train.imgs]

    print("Extracting features for val set...")
    
    test_features = []
    test_labels = []

    for samples, labels in data_loader_val:
        test_features.append(model(samples).detach().numpy())
        test_labels.append(labels.detach().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_filenames = [os.path.basename(f[0]) for f in dataset_val.imgs]
    
    
    for label in np.unique(test_labels):
        print(f"Class {label} has {np.sum(test_labels == label)} samples in the test set.")
    
    print("Features are ready!\nStart the classification.")
    
    os.makedirs(args.destination, exist_ok=True)
        
    # ============ logistic regression ... ============
    log_model = LogisticRegression(
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=args.seed,
    )
    
    log_model.fit(train_features, train_labels)
    
    probs = log_model.predict_proba(test_features)
    labs = np.zeros_like(probs)
    labs[np.arange(len(test_labels)), test_labels] = 1

    bins = np.linspace(0, 1, args.num_bins)
    step = bins[1] - bins[0]
    steps = bins - step / 2
    steps[0] = 0
    low = bins[:-1]
    upp = bins[1:]
            
    p = np.zeros(len(low))
    freqs = np.zeros(len(low))
    observed = np.zeros(len(low))
        
    for i in range(len(low)):
        _labs = labs[np.where((probs >= low[i]) * (probs < upp[i]))]
        p[i] = probs[np.where((probs >= low[i]) * (probs < upp[i]))].mean()
        freqs[i] = _labs.mean()
        observed[i] = _labs.shape[0]
    
    error = np.abs(freqs - p).mean()
    
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, np.log(observed), width=step*0.95, color="b")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observations (log)")
    plt.savefig(os.path.join(args.destination, "observations.png"), dpi=300)
    
    
    plt.figure(figsize=(10, 5))
    plt.bar((low + upp) / 2, freqs, width=step*0.95, color="b")
    plt.step(bins, np.concatenate([[0], p]), where="pre", color="k", linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration error: {error:.3f}")
    plt.savefig(os.path.join(args.destination, "calibration.png"), dpi=300)
    