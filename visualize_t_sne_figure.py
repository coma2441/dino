import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

feats = np.load('/Users/ima029/test/SCAMPI/Repository/scampi_unsupervised/frameworks/dino/trained_models/vit_small_fine_tuned/old/feats.npy')
labs = np.load('/Users/ima029/test/SCAMPI/Repository/scampi_unsupervised/frameworks/dino/trained_models/vit_small_fine_tuned/old/labels.npy')

t_sne = TSNE(n_components=2, random_state=0)
X_2d = t_sne.fit_transform(feats)
x = X_2d[:,0]
y = X_2d[:,1]

selected_classes = ['alisocysta', 'azolla', 'eatonicysta', 'inaperturopollenites', 'isabelidinium', 'svalbardella']

#colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

plt.figure(figsize=(12,8))
plt.scatter(x, y, c='black', alpha=0.2)
for i, c in enumerate(selected_classes):
    idx = np.where(labs == c)
    plt.scatter(x[idx], y[idx], label=c, alpha=0.8)
plt.legend()
plt.axis('off')
plt.savefig('t_sne.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
#plt.show()

