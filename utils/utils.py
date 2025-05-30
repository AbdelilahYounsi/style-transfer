import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size))
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def vgg_preprocess(batch):
    """
    VGG preprocessing: subtract mean pixel values
    """
    # VGG mean pixel values (BGR format, but RGB is close enough)
    mean = batch.new_tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    return batch - mean

def normalize_batch(batch):
    # Keep for backwards compatibility but not used in training
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


