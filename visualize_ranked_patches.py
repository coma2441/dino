# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import requests
from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import vision_transformer as vits


def get_embeddings(model, x):
    # get patch embeddings from x
    # output shape: (batch_size, num_patches, projection_dim)
    x = model.prepare_tokens(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize patch rankings')
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./ranked_patches/', help='Path where to save visualizations.')
    args = parser.parse_args()

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # =================================================================================
    # === BUILD MODEL =================================================================
    # =================================================================================
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
    # =================================================================================
    # === LOAD IMAGE ==================================================================
    # =================================================================================
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_tf = transform(img)
    img_tf = img_tf.unsqueeze(0) # add batch dimension
    img_tf = img_tf.to(device)
    # =================================================================================
    # === ANALYZE PATCHES =============================================================
    # =================================================================================
    embeddings = get_embeddings(model, img_tf) # shape (batch_size, num_patches, projection_dim)
    cls_token = embeddings[0, 0, :]  # CLS token embedding
    patch_tokens = embeddings[0, 1:, :]  # patch tokens, shape (num_patches, projection_dim)
    # affinitiy matrix
    A = torch.matmul(patch_tokens, patch_tokens.transpose(0, 1)) # shape (num_patches, num_patches)
    # normalize
    A = A * (A > 0)
    A = A / torch.sum(A, dim=(0))
    A = A.T
    
    for i in range(10):
        A = np.matmul(A, A)
    
    heatmap = A[0].reshape(28, 28)
    heatmap = Image.fromarray(np.array(heatmap)).resize((224, 224))
    heatmap = np.array(heatmap)
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap = 1 - heatmap
    
    original = np.array(img.resize((224, 224)))
    
    os.makedirs(args.output_dir, exist_ok=True)
    filename = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".png"
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(heatmap)
    axes[1].axis('off')
    axes[1].set_title('Heatmap')
    axes[2].imshow(original * np.repeat(heatmap[:, :, np.newaxis], 3, axis=2) / 255.)
    axes[2].axis('off')
    axes[2].set_title('Image * Heatmap')
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename), dpi=300)
