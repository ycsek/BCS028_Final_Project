import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def load_images(folder, label, max_images=1000):
    images = []
    labels = []
    for file in os.listdir(folder)[:max_images]:
        img_path = os.path.join(folder, file)
        img = Image.open(img_path).convert('L').resize((128, 128))
        img_array = np.array(img).flatten()  
        images.append(img_array)
        labels.append(label)
    return images, labels


normal_dir = "./COVID-19_Radiography_Dataset/Normal/images"
opacity_dir = "./COVID-19_Radiography_Dataset/Lung_Opacity/images"
covid_dir = "./COVID-19_Radiography_Dataset/Covid/images"
viral_pneumonia_dir = "./COVID-19_Radiography_Dataset/Viral Pneumonia/images"

dirs = [covid_dir, normal_dir, viral_pneumonia_dir, opacity_dir]
classes = ['COVID-19', 'Normal', 'Viral Pneumonia', 'Lung Opacity']
all_images = []
all_labels = []
for folder, cls in zip(dirs, classes):
    imgs, lbls = load_images(folder, cls)
    all_images.extend(imgs)
    all_labels.extend(lbls)

X = np.array(all_images)
y = np.array(all_labels)

tsne = TSNE(n_components=2, random_state=27, perplexity=30)
X_tsne = tsne.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=y_encoded, cmap='viridis', alpha=0.7)
cbar = plt.colorbar(scatter, ticks=range(len(le.classes_)),
                    format=plt.FuncFormatter(lambda val, loc: le.classes_[val]))
cbar.ax.tick_params(labelsize=18)
plt.title('t-SNE Visualization of COVID-19 Radiography Features', fontsize=22)
plt.xlabel('t-SNE Component 1', fontsize=20)
plt.ylabel('t-SNE Component 2', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('tsne_plot.png', dpi=300)
plt.show()
