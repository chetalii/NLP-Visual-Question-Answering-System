# interactive_demo.py
import os
from vqa_system import VQASystem

def interactive_demo():
    """
    Interactive demo where users can input their own images and questions
    """
    vqa_system = VQASystem(use_rule_based=True)
    
    print("=== Interactive VQA Demo ===")
    print("Enter 'quit' to exit")
    
    while True:
        # Get image path
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print("Image not found! Please check the path.")
            continue
        
        # Get question
        question = input("Enter your question about the image: ").strip()
        
        if question.lower() == 'quit':
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        # Process and display results
        try:
            results = vqa_system.process_input(image_path, question)
            print(f"\n🤖 Answer: {results['answer']}")
            print(f"📊 Analysis: {results['question_type']} question")
            print(f"🔑 Keywords: {', '.join(results['keywords'])}")
            
        except Exception as e:
            print(f"Error processing your request: {e}")

if __name__ == "__main__":
    interactive_demo()