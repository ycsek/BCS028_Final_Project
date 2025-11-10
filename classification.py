import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F


class RadiographyDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'COVID-19': 0, 'Normal': 1,
                             'Viral Pneumonia': 2, 'Lung Opacity': 3}
        dirs = [data_dirs['COVID-19'], data_dirs['Normal'],
                data_dirs['Viral Pneumonia'], data_dirs['Lung Opacity']]
        classes = ['COVID-19', 'Normal', 'Viral Pneumonia', 'Lung Opacity']
        for folder, cls in zip(dirs, classes):
            print(f"Loading data from {cls}")
            for file in tqdm(os.listdir(folder)):
                self.images.append(os.path.join(folder, file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dirs = {
    'Normal': "./COVID-19_Radiography_Dataset/Normal/images",
    'Lung Opacity': "./COVID-19_Radiography_Dataset/Lung_Opacity/images",
    'COVID-19': "./COVID-19_Radiography_Dataset/COVID/images",
    'Viral Pneumonia': "./COVID-19_Radiography_Dataset/Viral Pneumonia/images"
}
dataset = RadiographyDataset(data_dirs, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 20


for epoch in range(num_epochs):
    model.train()
    loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} "):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item()
    avg_loss = loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

model.eval()
y_true = []
y_pred = []
y_scores = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(probabilities.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
cm = confusion_matrix(y_true, y_pred)

print(f'Accuracy: {acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')


plt.figure(figsize=(14, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset.class_to_idx.keys(), yticklabels=dataset.class_to_idx.keys(),)
plt.title('Confusion Matrix', fontsize=22)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('True', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('cnn_cm.png', dpi=300)
plt.show()

n_classes = 4
y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(
        y_true_binarized[:, i], np.array(y_scores)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(14, 8))
colors = ['blue', 'green', 'red', 'cyan']
class_names = list(dataset.class_to_idx.keys())

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('ROC Curves', fontsize=22)
plt.legend(loc="lower right", fontsize=18)
plt.savefig('roc_curve.png', dpi=300)
plt.show()

for i in range(n_classes):
    print(f'AUC for class {class_names[i]}: {roc_auc[i]:.4f}')
