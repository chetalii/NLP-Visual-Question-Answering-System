# NLP-Visual-Question-Answering-System

A Python-based Visual Question Answering system that uses AI to answer questions about images. Built with Streamlit, Transformers, and OpenCV.

![VQA Demo](https://img.shields.io/badge/Project-VQA_System-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-green)
![Streamlit](https://img.shields.io/badge/Web_Framework-Streamlit-red)

## 🎯 Features

- **Image Upload**: Support for JPG, JPEG, PNG, BMP formats
- **Natural Language Questions**: Ask any question about uploaded images
- **AI-Powered Answers**: Uses ViLT transformer model for accurate responses
- **Real-time Processing**: Instant analysis and answers
- **User-friendly Interface**: Clean Streamlit web interface
- **Multiple Question Types**: Counting, color identification, object detection, scene description

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/visual-question-answering.git
   cd visual-question-answering

# Project Report: Visual Question Answering System

## 📋 Problem Statement

Visual Question Answering (VQA) is a challenging multimodal task that requires understanding both visual content and natural language. The goal was to build an AI system that can answer natural language questions about images, combining computer vision and NLP capabilities. This addresses the fundamental challenge of creating AI that can reason across different modalities (vision and language) to provide meaningful insights about visual content.

## 🗂️ Dataset & Resources

**Approach**: Used pre-trained models without requiring large custom datasets
- **ImageNet Pre-trained Models**: ResNet50 for feature extraction
- **VQA Pre-trained Model**: ViLT (Vision-and-Language Transformer) fine-tuned on VQA v2
- **NLP Models**: BERT for text understanding, NLTK for text processing
- **Computer Vision**: OpenCV with Haar Cascades for face detection
- **No additional training data required** - leverages transfer learning

## 🛠️ Methodology

### Final Architecture

**Hybrid VQA System** combining rule-based and neural approaches:

1. **Frontend Interface** (Streamlit):
   - Image upload with preview
   - Natural language question input
   - Real-time results display

2. **Backend Processing Pipeline**:
   ```
   Image + Question → ViLT Transformer → Answer Generation
   ```

3. **Primary AI Model**: ViLT (Vision-and-Language Transformer)
   - Model: `dandelin/vilt-b32-finetuned-vqa`
   - Jointly processes image and text embeddings
   - End-to-end VQA without separate feature extraction

4. **Fallback System**: Rule-based analyzer using:
   - OpenCV for color analysis and edge detection
   - Haar Cascades for face detection
   - Keyword matching for question classification

### Key Components

- **Image Processing**: OpenCV for basic analysis, ViLT for deep understanding
- **Text Processing**: BERT embeddings with question type classification
- **Multimodal Fusion**: ViLT transformer handles vision-language integration
- **Web Interface**: Streamlit for user-friendly interaction

## 📊 Results & Performance

### Capabilities
- ✅ **Accurate object counting** (people, general objects)
- ✅ **Color identification** and scene description
- ✅ **Existence queries** (people, objects, features)
- ✅ **Location analysis** and spatial understanding
- ✅ **Natural language understanding** of diverse question types

### Performance Metrics
- **Processing Time**: 2-5 seconds per query
- **Accuracy**: High for common objects and clear images
- **Model Size**: ~1GB (ViLT pre-trained weights)
- **Supported Image Formats**: JPG, JPEG, PNG, BMP

### Example Results
- Input: "How many people are in this photo?" → Output: "There are 4 people"
- Input: "What color is the main object?" → Output: "The main color is blue"
- Input: "Is this an outdoor scene?" → Output: "Yes, this appears to be outdoors"

## 🎯 Key Achievements

1. **Successfully implemented** a working VQA system with web interface
2. **Integrated state-of-the-art** transformer models (ViLT) for multimodal understanding
3. **Created user-friendly application** requiring no technical expertise
4. **Achieved reasonable accuracy** without extensive training data
5. **Built scalable architecture** that can incorporate improved models

## 🔮 Future Enhancements

- Integration with larger VQA datasets for improved accuracy
- Real-time video question answering capability
- Support for more complex reasoning questions
- Multi-language question support
- Mobile application deployment

This project demonstrates the practical implementation of multimodal AI systems, showcasing how computer vision and natural language processing can be combined to create intelligent applications that understand and reason about visual content through natural language interaction.
