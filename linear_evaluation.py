import os
import argparse

import numpy as np
import pandas as pd

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
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
    parser.add_argument('--img_size', default=224, type=int)
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
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size])
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
    
    #num_classes = np.unique(train_labels).shape[0]
    #class_names = ["class_" + str(i) for i in range(num_classes)]
    class_names = dataset_train.classes
    # table for class wise metrics
    table = pd.DataFrame()
    table["class"] = class_names
    # summary table for overall metrics
    summary_table = pd.DataFrame()
    
    # ============ logistic regression ... ============
    log_model = LogisticRegression(
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=args.seed,
    )
    
    log_model.fit(train_features, train_labels)
    
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=np.concatenate([train_labels, test_labels])
    )
    
    # compute the log loss on the test set
    y_pred = log_model.predict(test_features)
        
    summary_table.loc["log_model", "log_loss"] = log_loss(test_labels, log_model.predict_proba(test_features))
    
    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc["log_model", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
    summary_table.loc["log_model", "accuracy"] = accuracy_score(test_labels, y_pred)
    summary_table.loc["log_model", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")
    
    # add the metrics to the table
    table["log_model" + "_precision"] = precision_score(test_labels, y_pred, average=None)
    table["log_model" + "_recall"] = recall_score(test_labels, y_pred, average=None)
    table["log_model" + "_f1_score"] = f1_score(test_labels, y_pred, average=None)
    
    # save confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    cm_display = ConfusionMatrixDisplay(
        cm, display_labels=class_names
        )
    cm_display.plot(xticks_rotation="vertical", colorbar=False)
    cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)
    
    # also save predictions for each image
    preds = pd.DataFrame()
    preds["filename"] = test_filenames
    preds["true_label"] = test_labels
    preds["pred_label"] = y_pred
    preds.to_csv(os.path.join(args.destination, "predictions_" + "log_model" + ".csv"))

    # ============ k-NN ... ============
    for k in args.nb_knn:

        knn = KNeighborsClassifier(n_neighbors=k, p=2)
        knn.fit(train_features, train_labels)
        y_pred = knn.predict(test_features)
        
        # compute the accuracy and balanced accuracy and mean precision on the test set
        summary_table.loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
        summary_table.loc[f"k={k}", "accuracy"] = accuracy_score(test_labels, y_pred)
        summary_table.loc[f"k={k}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")
        
        # add the metrics to the table
        table[f"k={k}" + "_precision"] = precision_score(test_labels, y_pred, average=None)
        table[f"k={k}" + "_recall"] = recall_score(test_labels, y_pred, average=None)
        table[f"k={k}" + "_f1_score"] = f1_score(test_labels, y_pred, average=None)
        
    # save pandas table as csv to same folder as pretrained weights
    table.to_csv(os.path.join(args.destination, "class_wise_metrics.csv"))
    summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))
