# models/vqa_model.py
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import requests

class RealVQAModel:
    """
    A real VQA model using pre-trained ViLT (Vision-and-Language Transformer)
    This actually understands images and questions!
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading ViLT VQA model...")
        
        # Load pre-trained ViLT model - specifically trained for VQA
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model.to(self.device)
        self.model.eval()
        
        print("ViLT model loaded successfully!")
    
    def predict(self, image_path, question):
        """
        Use real AI to answer questions about images
        """
        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            encoding = self.processor(image, question, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()
                confidence = torch.softmax(logits, dim=-1).max().item()
            
            # Get predicted answer
            answer = self.model.config.id2label[predicted_class]
            
            return {
                'answer': answer,
                'confidence': round(confidence * 100, 2),
                'model': 'ViLT'
            }
            
        except Exception as e:
            print(f"Error in real VQA model: {e}")
            return {
                'answer': "I couldn't process that image and question",
                'confidence': 0,
                'model': 'ViLT'
            }

class HybridVQAModel:
    """
    Hybrid approach: Try real AI first, fallback to rule-based
    """
    def __init__(self):
        try:
            self.real_model = RealVQAModel()
            self.use_real_model = True
            print("Using real AI VQA model")
        except Exception as e:
            print(f"Failed to load real model: {e}")
            self.use_real_model = False
            print("Falling back to rule-based model")
            from models.vqa_model import EnhancedRuleBasedVQA
            self.rule_model = EnhancedRuleBasedVQA()
    
    def predict(self, image_analysis, question, question_type, image_path):
        if self.use_real_model:
            result = self.real_model.predict(image_path, question)
            return result['answer']
        else:
            # Fallback to rule-based
            return self.rule_model.predict(image_analysis, question, question_type, image_path)