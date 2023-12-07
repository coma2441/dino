import argparse

import torch

import tensorflow as tf
import pandas as pd

import utils
import vision_transformer as vits

from lora import LoRA_ViT_timm

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, cosine_distances

import numpy as np
import matplotlib.pyplot as plt



# following steps
# 1. Parse the record into tensors
# 2. Decode the image
# 3. Resize the image to 256x256 using bicubic interpolation
# 4. Center crop the image to 224x224 pixels
# 5. Normalize the image to [0,1]
# 6. Normalize the image using the mean and variance of the ImageNet dataset.


# ------------------ Load the pretrained ViT model ------------------
parser = argparse.ArgumentParser('Extract features using pretrained ViT models.')
parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights.")
parser.add_argument('--arch', default='vit_tiny', type=str, help='Architecture')
parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
args = parser.parse_args()

args.pretrained_weights = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/scampi_unsupervised/frameworks/dino/trained_models/vit_small_fine_tuned_lora_r8_p1920/checkpoint0009.pth"
args.arch = "vit_small"

model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
model = LoRA_ViT_timm(model, r=8)
print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
model.eval()


# ------------------ Load labelled data ------------------
def preprocess_image(image):
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.central_crop(image, central_fraction=224 / 256)
    image = tf.cast(image, tf.float32) / 255.
    #image = image - tf.constant([0.485, 0.456, 0.406])
    #image = image / tf.constant([0.229, 0.224, 0.225])
    return image

path_to_labelled = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/data/labelled crops"
path_to_csv = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/data/labelled crops/labels.csv"
csv_file = pd.read_csv(path_to_csv)
# only load images from class 1
files = csv_file[csv_file['label'] == 18]["filename"]
files = csv_file["filename"]

ds_labelled = tf.data.Dataset.from_tensor_slices(files)
ds_labelled = ds_labelled.map(lambda x: tf.io.read_file(path_to_labelled + "/" + x))
ds_labelled = ds_labelled.map(lambda x: tf.image.decode_jpeg(x, channels=3))
ds_labelled = ds_labelled.map(preprocess_image)


# ------------------ Extract features for labelled data ------------------
features = []
for image in ds_labelled:
    image = tf.expand_dims(image, axis=0)
    image = torch.from_numpy(image.numpy())
    image = image.permute(0, 3, 1, 2)
    with torch.no_grad():
        features.append(model(image).numpy())
features = np.array(features)
features = features.reshape(features.shape[0], features.shape[2])


# standardize features
features = features - features.mean(axis=0)
features = features / features.std(axis=0)

features

# ------------------ Load unlabelled data ------------------
def _tfrecord_map_function(x):
    """Parse a single image from a tfrecord file."""
    # Dict with key 'image' and value of type string
    x = tf.io.parse_single_example(x, {"image": tf.io.FixedLenFeature([], tf.string)})
    # Tensor of type uint8
    x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
    x = tf.image.encode_jpeg(x)
    x = tf.image.decode_jpeg(x, channels=3)
    return x

path_to_slide = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/data/NO 6407-6-5/tfrecords224x224/6407_6-5 1920 mDC.tfrecords"
ds_unlabelled = tf.data.TFRecordDataset(path_to_slide)
ds_unlabelled = ds_unlabelled.map(_tfrecord_map_function)
ds_unlabelled = ds_unlabelled.map(preprocess_image)


# ------------------ Extract features for unlabelled data ------------------
features_unlabelled = []
for image in ds_unlabelled:
    image = tf.expand_dims(image, axis=0)
    image = torch.from_numpy(image.numpy())
    image = image.permute(0, 3, 1, 2)
    with torch.no_grad():
        features_unlabelled.append(model(image).numpy())
features_unlabelled = np.array(features_unlabelled)
features_unlabelled = features_unlabelled.reshape(features_unlabelled.shape[0], features_unlabelled.shape[2])


features_unlabelled

from sklearn.cluster import KMeans


k_means = KMeans(n_clusters=1000, random_state=0).fit(features_unlabelled)

k_means.labels_


sim = euclidean_distances(features, features_unlabelled)


neighbors = np.argmin(sim, axis=1)


ims = np.array([im.numpy() for im in ds_labelled])

for i, im in enumerate(ds_unlabelled):
    if i in neighbors:
        #idx = np.where(neighbors == i)[0][0]
        idxs = np.where(neighbors == i)[0]
        #ims[idx] = im.numpy()
        for idx in idxs:
            ims[idx] = im.numpy()
    

# create folder
dest = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/lora_1920_neighbors"
import os
os.mkdir(dest)

for im, filename in zip(ims, files):
    # normalize image
    #im = im - im.min()
    #im = im / im.max()
    #im = im * 255
    # save image
    plt.imsave(dest + "/" + filename, im.astype(np.uint8))




import os
import matplotlib.pyplot as plt
import numpy as np

dest1 = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/pretrained_neighbors"

# create folder for each class
labels = np.unique(csv_file["label"])
for label in labels:
    os.mkdir(dest1 + "/" + str(label))

import shutil

# move images to corresponding folder
for filename in os.listdir(dest1):
    if ".jpg" not in filename:
        continue
    label = csv_file[csv_file["filename"] == filename]["label"].values[0]
    shutil.move(dest1 + "/" + filename, dest1 + "/" + str(label) + "/" + filename)
    


n = len(os.listdir(dest1))

ims1 = np.zeros((n, 224, 224, 3))
for i, filename in enumerate(os.listdir(dest1)):
    try:
        ims1[i] = plt.imread(dest1 + "/" + filename)
    except:
        print(filename)
        continue
    

neighbors[634]



dest2 = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/pretrained_neighbors"

ims2 = np.zeros((n, 224, 224, 3))

filenames = os.listdir(dest2)

for i, filename in enumerate(os.listdir(dest2)):
    try:
        ims2[i] = plt.imread(dest2 + "/" + filename)
    except:
        print(filename)
        continue

ims = np.array([im.numpy() for im in ds_labelled])


dest = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Desktop/SCAMPI/Repository/compare_neighbors"

os.mkdir(dest)

for filename, im1, im2 in zip(os.listdir(dest2), ims1, ims2):
    if not (im1 == im2).all():
        try:
        
            labelled_crop = plt.imread(path_to_labelled + "/" + filename)
            
            fig, axs = plt.subplots(1, 3, figsize=(10, 10))
            axs[0].imshow(labelled_crop)
            axs[0].set_title("labelled crop")
            axs[0].axis('off')
            axs[1].imshow(im1.astype(np.uint8))
            axs[1].set_title("fine-tuned")
            axs[1].axis('off')
            axs[2].imshow(im2.astype(np.uint8))
            axs[2].set_title("pretrained")
            axs[2].axis('off')
            plt.tight_layout()
            plt.savefig(dest + "/" + filename)
        
        except OSError:
            print(filename)
            continue

label_counts = np.unique(csv_file["label"], return_counts=True)

labels = {label:0 for label in label_counts[0]}

for filename, im1, im2 in zip(os.listdir(dest2), ims1, ims2):
    is_equal = (im1 == im2).all()
    labels[csv_file[csv_file["filename"] == filename]["label"].values[0]] += is_equal

labels = {k: v / label_counts[1][k] for k, v in labels.items()}

ims = []
idxs = []

for i, im in enumerate(ds_unlabelled):
    if i in neighbors:
        ims.append(im.numpy())
        idxs.append(i)

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(ims[i])
    ax.axis('off')
plt.tight_layout()
plt.show()




sim = manhattan_distances(features, features_unlabelled).mean(axis=0)
sim = cosine_distances(features, features_unlabelled).mean(axis=0)

ds_sim = tf.data.Dataset.from_tensor_slices(sim)
ds_zip = tf.data.Dataset.zip((ds_unlabelled, ds_sim))

ds_lab = tf.data.Dataset.from_tensor_slices(k_means.labels_)
ds_zip = tf.data.Dataset.zip((ds_unlabelled, ds_lab))


(sim < 0.046).sum()

sim.min()

ims = []
d = []

for im, s in ds_zip:
    if s < 0.046:
        ims.append(im.numpy())
        d.append(s.numpy())


fig, axs = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
    ax.imshow(ims[np.argsort(d)[i]])
    ax.set_title(d[i])
    ax.axis('off')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression

from networkx import Graph
import networkx as nx

from sklearn.neighbors import kneighbors_graph



distance = euclidean_distances(features_unlabelled, features_unlabelled)

start = neighbors[8]
start = 13847

idxs = [start]
distances = [0]

for i in range(99):
    nns = distance[idxs[-1]].argsort()[:100]
    for nn in nns:
        if nn not in idxs:
            idxs.append(nn)
            distances.append(distance[idxs[-2], nn])
            break

csv_file.iloc[8]


idxs = distance[1149].argsort()[:100]

#ims = []
#distances = []
ims = np.zeros((100, 224, 224, 3))

for i, im in enumerate(ds_unlabelled):
    if i in idxs:
        #ims.append(im.numpy())
        ims[idxs.index(i)] = im.numpy()
        #distances.append(distance[1149, i])
        

ims = np.array(ims)
distances = np.array(distances)
# sort by distance
ims = ims[np.argsort(distances)]
distances = distances[np.argsort(distances)]

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(ims[i].astype(np.uint8))
    ax.set_title(distances[i])
    ax.axis('off')
plt.tight_layout()
plt.show()


feats = features_unlabelled[idxs]

from sklearn.manifold import TSNE, MDS

mds = MDS(n_components=2, verbose=1, max_iter=1000)





knn_graph = kneighbors_graph(features_unlabelled, n_neighbors=4, mode='connectivity', include_self=True)
# graph from knn of features
g = Graph(knn_graph)        
#
distance = euclidean_distances(features_unlabelled, features_unlabelled)
# replace edge weights with distance
for i, j in g.edges():
    g[i][j]['weight'] = distance[i, j]




# find shortest paths for 2602
shortest_paths = nx.shortest_path_length(g, source=2602, weight='weight')



test = list(shortest_paths.keys())[:100]
lenghts = list(shortest_paths.values())[:100]

ims = []
lengths = []

for i, im in enumerate(ds_unlabelled):
    if i in test:
        ims.append(im.numpy())
        lengths.append(lenghts[test.index(i)])

ims = np.array(ims)
lengths = np.array(lengths)
# sort by length
ims = ims[np.argsort(lengths)]
lengths = lengths[np.argsort(lengths)]
    
fig, axs = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
    ax.imshow(ims[i])
    ax.set_title(lengths[i])
    ax.axis('off')
plt.tight_layout()
plt.show()


a, b = np.unique(np.argmin(sim, axis=1), return_counts=True)
a[b == 2]


from sklearn.manifold import TSNE, MDS

t_sne = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=1)
t_sne.fit(features_unlabelled)

mds = MDS(n_components=2, verbose=1, max_iter=1000)
mds.fit(features_unlabelled)

plt.scatter(mds.embedding_[:, 0], mds.embedding_[:, 1], c=k_means.labels_)
plt.legend()
plt.show()


plt.scatter(t_sne.embedding_[:, 0], t_sne.embedding_[:, 1], c=k_means.labels_)
# annotate points with labels
for i, txt in enumerate(k_means.labels_):
    plt.annotate(txt, (t_sne.embedding_[i, 0], t_sne.embedding_[i, 1]))
plt.legend()
plt.show()



(k_means.labels_ == 24).sum()
counter = 0
for im, s in ds_zip:
    if s == 49:
        plt.imshow(im.numpy())
        plt.show()
        counter += 1
    if counter == 30:
        break


d = []

for i in range(1000):
    j = sim[k_means.labels_ == i].mean()
    
    d.append(j)

d = np.array(d)

(k_means.labels_ == 242).sum()

np.argsort(d)

for im, lab in ds_zip:
    if lab == 163:
        plt.imshow(im.numpy())
        plt.show()
        