import os
import sys
import shutil
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.distributed as dist
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from lora import LoRA_ViT_timm


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    
    sampler = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    #model.cuda()
    if args.lora_rank is not None:
        model = LoRA_ViT_timm(model, r=args.lora_rank)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    
    # create a temporary folder to store features
    # Note: this is a hack to avoid OOM when extracting features
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = os.path.join("features", time_stamp)
    os.makedirs(destination)
    
    for i, (samples, labels) in enumerate(data_loader_train):
        feats = model(samples)
        
        torch.save(feats.cpu(), os.path.join(destination, "trainfeat_" + str(i) + ".pth"))
        torch.save(labels.cpu(), os.path.join(destination, "trainlabels_" + str(i) + ".pth"))
    
    print("Extracting features for val set...")
    
    for i, (samples, labels) in enumerate(data_loader_val):
        feats = model(samples)

        torch.save(feats.cpu(), os.path.join(destination, "testfeat_" + str(i) + ".pth"))
        torch.save(labels.cpu(), os.path.join(destination, "testlabels_" + str(i) + ".pth"))

    files = os.listdir(destination)
    files = [f for f in files if f.endswith('.pth')]
    files = sorted(files)

    test_features = [f for f in files if f.startswith('testfeat')]
    test_labels = [f for f in files if f.startswith('testlabel')]
    train_features = [f for f in files if f.startswith('trainfeat')]
    train_labels = [f for f in files if f.startswith('trainlabel')]

    test_features = [torch.load(os.path.join(destination, f)) for f in test_features]
    test_labels = [torch.load(os.path.join(destination, f)) for f in test_labels]
    train_features = [torch.load(os.path.join(destination, f)) for f in train_features]
    train_labels = [torch.load(os.path.join(destination, f)) for f in train_labels]

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    # delete temporary folder and contents
    shutil.rmtree(destination)  
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=3, type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=False, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_tiny', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='./labelled crops 20', type=str, help='Path to evaluation dataset')
    
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)
    
    train_features = train_features.detach().numpy()
    train_labels = train_labels.detach().numpy()
    test_features = test_features.detach().numpy()
    test_labels = test_labels.detach().numpy()    
    
    print("Features are ready!\nStart the classification.")
    
    model_id = args.arch + "_" + str(args.patch_size) + "x" + str(args.patch_size)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    num_classes = np.unique(train_labels).shape[0]
    class_names = ["class_" + str(i) for i in range(num_classes)]
    
    # table for class wise metrics
    table = pd.DataFrame()
    table["class"] = class_names
    # summary table for overall metrics
    summary_table = pd.DataFrame()
    summary_table.index = [model_id]
    summary_table["model_id"] = model_id
    
    # ============ logistic regression ... ============
    
    # train a logistic regression model and compute the log loss
    log_model = LogisticRegression(
        max_iter=10000,
        multi_class="multinomial",
        class_weight="balanced",
    )
    
    log_model.fit(train_features, train_labels)
    
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=np.concatenate([train_labels, test_labels])
    )
    sample_weights = np.array([class_weights[y] for y in test_labels])
    
    # compute the log loss on the test set
    y_pred = log_model.predict(test_features)
        
    summary_table.loc[model_id, "log_loss"] = log_loss(
        test_labels, log_model.predict_proba(test_features)
    )
    
    # ...the same for the training set
    y_pred_train = log_model.predict(train_features)

    summary_table.loc[model_id, "log_loss_train"] = log_loss(
        train_labels, log_model.predict_proba(train_features)
    )

    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc[model_id, "log_balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
    summary_table.loc[model_id, "log_accuracy"] = accuracy_score(test_labels, y_pred)
    summary_table.loc[model_id, "log_mean_precision"] = precision_score(test_labels, y_pred, average="macro")

    # ...the same for the training set
    summary_table.loc[model_id, "log_balanced_accuracy_train"] = balanced_accuracy_score(train_labels, y_pred_train)
    summary_table.loc[model_id, "log_accuracy_train"] = accuracy_score(train_labels, y_pred_train)
    summary_table.loc[model_id, "log_mean_precision_train"] = precision_score(train_labels, y_pred_train, average="macro")

    # add the metrics to the table
    table[model_id + "_log_precision"] = precision_score(test_labels, y_pred, average=None)
    table[model_id + "_log_recall"] = recall_score(test_labels, y_pred, average=None)

    # ============ k-NN ... ============
    
    # train a k-nearest neighbors model
    knn = KNeighborsClassifier(n_neighbors=args.nb_knn, p=2)
    knn.fit(train_features, train_labels)
    y_pred = knn.predict(test_features)
    y_pred_train = knn.predict(train_features)

    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc[model_id, "knn_balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
    summary_table.loc[model_id, "knn_accuracy"] = accuracy_score(test_labels, y_pred)
    summary_table.loc[model_id, "knn_mean_precision"] = precision_score(test_labels, y_pred, average="macro")

    # ...the same for the training set
    summary_table.loc[model_id, "knn_balanced_accuracy_train"] = balanced_accuracy_score(train_labels, y_pred_train)
    summary_table.loc[model_id, "knn_accuracy_train"] = accuracy_score(train_labels, y_pred_train)
    summary_table.loc[model_id, "knn_mean_precision_train"] = precision_score(train_labels, y_pred_train, average="macro")

    # add the metrics to the table
    table[model_id + "_knn_precision"] = precision_score(test_labels, y_pred, average=None)
    table[model_id + "_knn_recall"] = recall_score(test_labels, y_pred, average=None)

    # save pandas table as csv to same folder as pretrained weights
    folder = args.pretrained_weights.rsplit("/", 1)[0]
    table.to_csv(os.path.join(folder, args.pretrained_weights.split("/")[-1].split(".")[0] + "_table_" + ".csv"))
    summary_table.to_csv(os.path.join(folder, args.pretrained_weights.split("/")[-1].split(".")[0] + "_summary_table_" + ".csv"))
    