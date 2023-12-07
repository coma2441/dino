import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE, MDS

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits


parser = argparse.ArgumentParser('Extract features using pretrained ViT models.')
parser.add_argument('--pretrained_weights', default='./trained_models/vit_small_fine_tuned/checkpoint0002.pth', type=str, help="Path to pretrained weights.")
parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument("--data_path", default="./labelled crops", type=str, help='Path to data.')
parser.add_argument("--method", default="mds", type=str, help='Method to use for dimensionality reduction.')
parser.add_argument("--destination", default="./trained_models/vit_small_fine_tuned/figures", type=str, help='Path to save the embeddings.')
args = parser.parse_args()

args.pretrained_weights = ""

if __name__ == '__main__':

    # ------------------ Load the pretrained ViT model ------------------

    print(f"Building {args.arch} model")
    
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    #model = LoRA_ViT_timm(model, r=8)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    
    # ------------------ Load the data ------------------
    
    print(f"Loading data from {args.data_path}")
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    
    # ------------------ Extract features ------------------
    
    print("Extracting features")
    
    features = []
    labels = []
    class_names = []
    
    for i, (image, label) in enumerate(dataset):
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            features.append(model(image).numpy())
            class_name = dataset.classes[label]
            name = class_name.rsplit("_")[-1]
            label = class_name.split("_")[0] # label sorted by class count
            labels.append(int(label))
            class_names.append(name)
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[2])
    labels = np.array(labels)
    class_names = np.array(class_names)
    
    # ------------------ Reduce dimensionality ------------------
    
    print("Reducing dimensionality")
    
    if args.method == "tsne":
        tsne = TSNE(n_components=2, random_state=0)
        features = tsne.fit_transform(features)
    elif args.method == "mds":
        mds = MDS(n_components=2, random_state=0)
        features = mds.fit_transform(features)
    else:
        raise ValueError("Method not supported.")
    
    # ------------------ Plot ------------------
    
    print("Plotting")
    
    destination = f"{args.destination}/{args.method}"
    
    os.makedirs(destination, exist_ok=True)
    
    for label in np.unique(labels):
        plt.figure()
        plt.scatter(features[labels != label, 0], features[labels != label, 1], alpha=0.2)
        plt.scatter(features[labels == label, 0], features[labels == label, 1], label=f"{label} ({name})", alpha=0.5)
        # change legend position
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.axis("off")
        plt.savefig(f"{destination}/{label}_{name}.png")






from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=100, random_state=0)
k_means.fit(features)

k_means.labels_

from sklearn.metrics import rand_score

rand_score(labels, k_means.labels_)

a, b = np.unique(k_means.labels_, return_counts=True)

c = a[b.argsort()[-10:]]

for i in c:
    print(f"{i}: {b[i]}")


for i in c:
    cluster = i

    c_feats = []
    ims = []
    for i in np.where(k_means.labels_ == cluster)[0]:
        c_feats.append(features[i])
        image = dataset[i][0]
        # unnormalize image
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image.permute(1, 2, 0)
        ims.append(image)

    mds = MDS(n_components=2, random_state=0)
    c_feats = np.array(c_feats)
    mds_feats = mds.fit_transform(c_feats)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    plt.figure()
    plt.scatter(mds_feats[:, 0], mds_feats[:, 1], alpha=0.2)
    for i in range(len(ims)):
        imagebox = OffsetImage(ims[i], zoom=0.2)
        ab = AnnotationBbox(imagebox, mds_feats[i], frameon=True)
        plt.gca().add_artist(ab)
    plt.axis("off")
    plt.show()