import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from PIL import Image


def weights_viz(weights, index):
    weights = torch.sub(weights, torch.min(weights))
    weights = torch.mul(weights, torch.div(255, torch.max(weights)))
    weights = weights.detach().numpy()[0]
    image = []
    for i in range(0, 50176, 224):
        image.append(weights[i:i + 224])
    image = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.savefig("weights_img/weights{}.png".format(index))


def img_to_tensor(path, crop=False):
    if not crop:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([373, 373]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    return tensor

