import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

model = mobilenet_v2(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

normal_dir = "COVID-19_Radiography_Dataset/Normal/images"
covid_dir = "COVID-19_Radiography_Dataset/Covid/images"
viral_pneumonia_dir = "COVID-19_Radiography_Dataset/Viral Pneumonia/images"
opacity_dir = "COVID-19_Radiography_Dataset/Lung_Opacity/images"

categories = ['Normal', 'COVID-19', 'Viral Pneumonia', 'Lung Opacity']
dirs = [normal_dir, covid_dir, viral_pneumonia_dir, opacity_dir]


num_images = 1000
mean_intensities = []

for label, dir_path in zip(categories, dirs):
    files = [f for f in os.listdir(
        dir_path) if f.endswith('.png')][:num_images]
    if not files:
        mean_intensities.append([])
        continue

    print(f"Processing {label} samples")
    category_means = []
    for filename in tqdm(files):
        image_path = os.path.join(dir_path, filename)
        img = Image.open(image_path).convert('L') 
        img_array = np.array(img)
        mean_intensity = np.mean(img_array)
        category_means.append(mean_intensity)

    mean_intensities.append(category_means)

plt.figure(figsize=(12, 8))
positions = range(len(categories))
plt.yticks(fontsize=18)
plt.violinplot(mean_intensities, positions=positions,
               showmeans=True, showmedians=True)
plt.xticks(positions, categories, fontsize=18)
plt.title('Violin Plot of Mean Pixel Intensities by Category', fontsize=26)
plt.xlabel('Category', fontsize=24)
plt.ylabel('Mean Pixel Intensity', fontsize=24)
plt.grid(True)
plt.savefig('violin_plot.png', dpi=600)
plt.show()
