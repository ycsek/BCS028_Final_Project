import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import scipy.ndimage as ndimage

# Define GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = []
        self.gradients = []
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation))
        self.handles.append(
            target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, x, class_idx=None):
        self.activations = []
        self.gradients = []
        logit = self.model(x)
        if class_idx is None:
            class_idx = logit.argmax(dim=1).item()
        score = logit[:, class_idx].sum()
        self.model.zero_grad()
        score.backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        alpha = gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((alpha * activations).sum(dim=1, keepdim=True))
        cam = cam / (cam.max() + 1e-6)
        return cam, logit, class_idx

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


# Load the pre-trained model
model = mobilenet_v2(pretrained=True)
model.eval()
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# data paths
normal_dir = "./COVID-19_Radiography_Dataset/Normal/images"
covid_dir = "./COVID-19_Radiography_Dataset/Covid/images"
viral_pneumonia_dir = "./COVID-19_Radiography_Dataset/Viral Pneumonia/images"
opacity_dir = "./COVID-19_Radiography_Dataset/Lung_Opacity/images"

categories = ['Normal', 'COVID-19', 'Viral Pneumonia', 'Lung Opacity']
dirs = [normal_dir, covid_dir, viral_pneumonia_dir, opacity_dir]

# Visualization
plt.figure(figsize=(14, 12))

for i, (label, dir_path) in enumerate(zip(categories, dirs)):
    # Get the first PNG file
    files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
    if not files:
        continue
    filename = files[6]
    image_path = os.path.join(dir_path, filename)

    img = Image.open(image_path).convert('RGB')
    orig_img = np.array(img.resize((224, 224))) / 255.0
    input_tensor = transform(img).unsqueeze(0)

    cam_tensor, logit, class_idx = gradcam(input_tensor)
    cam = cam_tensor[0, 0].cpu().numpy()
    cam = ndimage.zoom(cam, (224 / cam.shape[0], 224 / cam.shape[1]), order=3)

    heatmap = np.uint8(255 * plt.cm.jet(cam)[..., :3])
    superimposed = orig_img * 0.6 + (heatmap / 255.0) * 0.4

    # Original
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(orig_img)
    plt.title(f'{label}', fontsize=14)
    plt.axis('off')

    # Heatmap
    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM', fontsize=14)
    plt.axis('off')

    # Superimposed
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(superimposed)
    plt.title('Superimposed', fontsize=14)
    plt.axis('off')
plt.tight_layout()
plt.savefig('grad_cam.png', dpi=600)
plt.show()

gradcam.remove_hooks()
