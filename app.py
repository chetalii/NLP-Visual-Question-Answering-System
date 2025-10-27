# app.py
import streamlit as st
import os
from PIL import Image
import tempfile
import sys
import numpy as np

# Add the project root to Python path
sys.path.append('.')

# Import your VQA system
from vqa_system import VQASystem

# Set page configuration
st.set_page_config(
    page_title="Visual Question Answering System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def initialize_vqa_system():
    """Initialize the VQA system"""
    try:
        return VQASystem()  # ← REMOVED the use_rule_based parameter
    except Exception as e:
        st.error(f"Error initializing VQA system: {e}")
        return None

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🔍 Visual Question Answering System</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about any image and get AI-powered answers!")
    
    # Brief instructions
    st.markdown("""
    **How to use:**
    1. Upload an image below
    2. Ask a question about the image
    3. Get instant AI-powered answers!
    """)

    # Initialize VQA system
    with st.spinner("Loading AI models... This may take a moment."):
        vqa_system = initialize_vqa_system()

    if vqa_system is None:
        st.error("Failed to initialize the VQA system. Please check the backend setup.")
        return

    # Image upload section
    st.markdown("---")
    st.markdown('<div class="sub-header">📤 Upload Your Image</div>', unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image file (JPEG, PNG, BMP)"
    )

    # Display uploaded image
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_image_path = tmp_file.name

            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.success(f"✅ Image uploaded successfully! Size: {image.size}")

        except Exception as e:
            st.error(f"Error processing image: {e}")
            temp_image_path = None
    else:
        temp_image_path = None
        st.info("👆 Please upload an image to get started")

    # Question section
    st.markdown("---")
    st.markdown('<div class="sub-header">❓ Ask Your Question</div>', unsafe_allow_html=True)
    
    # Example questions as buttons
    st.markdown("**Try these example questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👥 Count Objects", use_container_width=True):
            st.session_state.demo_question = "How many people are in this photo?"
    
    with col2:
        if st.button("🎨 Identify Colors", use_container_width=True):
            st.session_state.demo_question = "What color is the main object?"
    
    with col3:
        if st.button("🔍 Describe Scene", use_container_width=True):
            st.session_state.demo_question = "What is happening in this picture?"

    # Question input
    question = st.text_area(
        "Enter your question about the image:",
        placeholder="e.g., How many people are in this photo? What color is the main object?",
        height=100,
        value=st.session_state.get('demo_question', '')
    )

    # Process button
    process_clicked = st.button(
        "🚀 Get Answer",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or not question.strip())
    )

    # Results section
    if process_clicked and temp_image_path and question.strip():
        with st.spinner("🤔 Analyzing image and processing question..."):
            try:
                # Process the question
                results = vqa_system.process_input(temp_image_path, question)
                
                st.markdown("---")
                st.markdown("### 📋 Analysis Results")
                
                # Answer in a nice box
                st.markdown(f"""
                <div class="answer-box">
                    <h3 style="color: #1f77b4; margin-top: 0;">Answer:</h3>
                    <p style="font-size: 1.5rem; font-weight: bold;">{results['answer']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Additional analysis details
                with st.expander("📊 View Detailed Analysis"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Question Type", results['question_type'].title())
                    
                    with col_b:
                        st.metric("Keywords", ", ".join(results['keywords']))
                    
                    with col_c:
                        brightness = results['image_analysis'].get('brightness', 0)
                        st.metric("Image Brightness", f"{brightness:.1f}")

                    # Image analysis details
                    st.subheader("Image Analysis")
                    if 'shape' in results['image_analysis']:
                        st.write(f"**Image Dimensions:** {results['image_analysis']['shape']}")

            except Exception as e:
                st.error(f"Error processing your request: {e}")
                st.info("Please try a different image or question.")

        # Clean up temporary file
        try:
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        except:
            pass

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit • Visual Question Answering System"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()