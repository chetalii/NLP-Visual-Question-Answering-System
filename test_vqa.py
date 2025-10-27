# test_vqa.py
import unittest
from vqa_system import VQASystem
from utils.text_processor import TextProcessor

class TestVQASystem(unittest.TestCase):
    
    def setUp(self):
        self.vqa_system = VQASystem()
        self.text_processor = TextProcessor()
    
    def test_question_typing(self):
        test_cases = [
            ("how many people are there?", "counting"),
            ("what color is the car?", "color"), 
            ("is there a dog in the image?", "existence"),
            ("where is the book?", "location")
        ]
        
        for question, expected_type in test_cases:
            detected_type = self.text_processor.identify_question_type(question)
            self.assertEqual(detected_type, expected_type)
    
    def test_keyword_extraction(self):
        question = "How many red cars are in this picture?"
        keywords = self.text_processor.extract_keywords(question)
        
        self.assertIn('red', keywords)
        self.assertIn('cars', keywords)
        self.assertNotIn('how', keywords)

if __name__ == '__main__':
    unittest.main()