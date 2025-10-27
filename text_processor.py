# utils/text_processor.py
from transformers import BertTokenizer, BertModel
import torch
import re
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt')
except:
    print("NLTK punkt not available, using simple tokenization")

class TextProcessor:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def preprocess_question(self, question):
        """
        Clean and preprocess the question
        """
        # Convert to lowercase
        question = question.lower()
        
        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        
        return question
    
    def extract_question_embeddings(self, question):
        """
        Extract embeddings from question using BERT
        """
        # Preprocess question
        cleaned_question = self.preprocess_question(question)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            cleaned_question, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        return embeddings.squeeze().numpy()
    
    def identify_question_type(self, question):
        """
        Identify the type of question for better answering
        """
        question_lower = question.lower()
        
        # Question type mapping
        question_types = {
            'counting': ['how many', 'count', 'number of'],
            'color': ['what color', 'color', 'colour'],
            'existence': ['is there', 'are there', 'does', 'do', 'has'],
            'location': ['where', 'which location', 'position'],
            'description': ['what is', 'describe', 'what does']
        }
        
        for q_type, keywords in question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
        
        return 'general'
    
    def extract_keywords(self, question):
        """
        Extract important keywords from question
        """
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'where', 'how', 'why', 'when'}
        tokens = word_tokenize(question.lower())
        keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        return keywords