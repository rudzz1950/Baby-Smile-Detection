"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Testing
"""
import numpy as np 
import argparse
import logging
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True
import cv2
import torchvision.transforms.transforms as transforms

from model.model import Mini_Xception
from utils import get_label_emotion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/weights_epoch_75.pth.tar')
    args = parser.parse_args()
    return args

# Function to load and preprocess an image
def load_image(image_path, target_size=(48, 48)):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image
    resized_image = cv2.resize(image, target_size)
    # Convert image to tensor and normalize
    transformed_image = transforms.ToTensor()(resized_image)
    return transformed_image

# Function to perform inference on an image
def test_image(image_path, model, device):
    # Load and preprocess the image
    image = load_image(image_path)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        _, predicted_label = torch.max(output, 1)
    
    # Get predicted emotion label
    predicted_emotion = get_label_emotion(predicted_label.item())
    return predicted_emotion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # Load pre-trained model
    model = Mini_Xception()
    model.to(device)
    model.eval()

    # Load pre-trained weights
    checkpoint = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(checkpoint['mini_xception'], strict=False)
    print(f'\tLoaded checkpoint from {args.pretrained}\n')

    # Input image path
    image_path = "/content/baby_smile_0703_58816_face_0.jpg"

    # Perform inference on the input image
    predicted_emotion = test_image(image_path, model, device)
    print(f'Predicted emotion: {predicted_emotion}')

if __name__ == "__main__":
    main()
