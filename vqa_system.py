# vqa_system.py (updated imports and class)
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.image_processor import ImageProcessor
from utils.text_processor import TextProcessor
from models.vqa_model import HybridVQAModel  # Changed import

class VQASystem:
    def __init__(self, use_real_model=True):  # Changed parameter
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        
        # Always use hybrid model now
        self.vqa_model = HybridVQAModel()
        print("VQA System initialized with AI model!")
    
    def process_input(self, image_path, question):
        """
        Process image and question to generate answer
        """
        print(f"Processing image: {image_path}")
        print(f"Question: {question}")
        
        # Step 1: Process image (for display purposes)
        image_analysis = self.image_processor.simple_image_analysis(image_path)
        
        # Step 2: Process question (for display purposes)
        question_type = self.text_processor.identify_question_type(question)
        keywords = self.text_processor.extract_keywords(question)
        
        print(f"Question type: {question_type}")
        print(f"Keywords: {keywords}")
        
        # Step 3: Generate answer using REAL AI
        answer = self.vqa_model.predict(image_analysis, question, question_type, image_path)
        
        # Step 4: Return results
        results = {
            'answer': answer,
            'question_type': question_type,
            'keywords': keywords,
            'image_analysis': {
                'shape': image_analysis.get('shape', 'unknown'),
                'brightness': round(image_analysis.get('brightness', 0), 2)
            }
        }
        
        return results
    
    def display_results(self, image_path, question, results):
        """
        Display the image, question, and answer
        """
        plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(2, 2, 1)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')
        
        # Display question and answer
        plt.subplot(2, 2, 2)
        plt.text(0.1, 0.8, f"Question: {question}", fontsize=12, wrap=True)
        plt.text(0.1, 0.6, f"Answer: {results['answer']}", fontsize=14, weight='bold', wrap=True)
        plt.text(0.1, 0.4, f"Type: {results['question_type']}", fontsize=10, wrap=True)
        plt.text(0.1, 0.2, f"Keywords: {', '.join(results['keywords'])}", fontsize=10, wrap=True)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()