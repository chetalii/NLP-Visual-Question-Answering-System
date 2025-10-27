# utils/image_processor.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        # Load pre-trained ResNet model for image features
        self.model = models.resnet50(pretrained=True)
        self.model.eval()  # Set to evaluation mode
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_path):
        """
        Extract features from image using pre-trained CNN
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            return features.squeeze().numpy()
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def simple_image_analysis(self, image_path):
        """
        Simple image analysis for demo purposes
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        analysis = {
            'shape': image.shape,
            'colors_detected': self.detect_dominant_colors(image),
            'edges_detected': self.detect_edges(image),
            'brightness': np.mean(image)
        }
        
        return analysis
    
    def detect_dominant_colors(self, image, k=3):
        """
        Simple color detection using k-means
        """
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers.astype(int)
    
    def detect_edges(self, image):
        """
        Detect edges in image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0)  # Count edge pixels