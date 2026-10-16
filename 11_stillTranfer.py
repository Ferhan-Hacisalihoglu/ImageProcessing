from pathlib import Path

import torch
import numpy as np
from torch import nn,optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, max_size=256,shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if shape is not None:
        size = shape
    else:
        size = max(image.size)
        if size > max_size:
            size = max_size

    in_transform = transforms.Compose([
        transforms.Resize((size) if isinstance(size, int) else size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0).numpy()
    # denormalize
    image = image * np.array((0.229, 0.224, 0.225))[:, None, None]
    image = image + np.array((0.485, 0.456, 0.406))[:, None, None]
    image = np.clip(image, 0, 1)
    return image.transpose(1, 2, 0)

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2", 
            "28": "conv5_1"
        }

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def run_style_transfer(content_image, style_image, step = 100,style_weight=1e6, content_weight=1):
    target =  content_image.clone().requires_grad_(True).to(device)
    
    optimizer = optim.Adam([target], lr=0.003)
    model = VGGFeatures().to(device)

    for step in tqdm(range(step)):
        target_features = model(target)
        content_features = model(content_image)
        style_features = model(style_image)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_features[layer])
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step [{step}/{step}], Total Loss: {total_loss.item():.4f}")

    return target

data_dir = Path(__file__).resolve().parent / "data" / "neuralStyleTransferData"

content = load_image(data_dir / "YellowLabradorLooking_new.jpg")
style = load_image(
    data_dir / "Vassily_Kandinsky,_1913_-_Composition_7.jpg",
    shape=content.shape[-2:],
)

output = run_style_transfer(content, style)

plt.figure(figsize=(10,5))
plt.imshow(im_convert(output))
plt.title("Stil transfer result")
plt.axis('off')
plt.show()


    