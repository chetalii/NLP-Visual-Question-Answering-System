# test_enhanced.py
from models.vqa_model import EnhancedRuleBasedVQA
import cv2
import requests
from PIL import Image
import os

def download_sample_images():
    """Download some better test images"""
    urls = [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Person
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # Nature
        "https://images.unsplash.com/photo-1464822759844-4c9a8537f727?w=400",  # Mountains
    ]
    
    os.makedirs('test_images', exist_ok=True)
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, stream=True)
            with open(f'test_images/sample_{i}.jpg', 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded sample_{i}.jpg")
        except:
            print(f"Couldn't download image {i}")

def test_enhanced_system():
    """Test the enhanced VQA system"""
    vqa = EnhancedRuleBasedVQA()
    
    # Use any images you have
    test_images = [
        'test_images/sample_0.jpg',  # Person image
        'test_images/sample_1.jpg',  # Nature image  
        'test_images/sample_2.jpg',  # Landscape
    ]
    
    test_questions = [
        "How many people are in this photo?",
        "What color is the main object?",
        "Is there a person in this image?",
        "Where is the main subject?",
        "Describe this image"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Analyzing: {image_path}")
            print(f"{'='*50}")
            
            for question in test_questions:
                # Simple analysis for the enhanced predictor
                analysis = {'shape': 'unknown', 'brightness': 0}
                answer = vqa.predict(analysis, question, 'general', image_path)
                print(f"Q: {question}")
                print(f"A: {answer}")
                print()
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    print("Downloading sample images...")
    download_sample_images()
    print("\nTesting enhanced VQA system...")
    test_enhanced_system()