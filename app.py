"""
Streamlit Web App for Brain Tumor Detection

This script creates a web interface for the brain tumor detection model.
It allows users to upload an image and get predictions.

Usage: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model
from brain_tumor_detection import crop_brain_contour

def predict_tumor(model, image_array):
    """
    Predict whether an image contains a tumor
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        
    Returns:
        prediction: 0 (no tumor) or 1 (tumor)
        probability: Probability of the prediction
    """
    # Get prediction probability
    pred_prob = model.predict(image_array, verbose=0)[0][0]
    
    # Convert to binary prediction
    prediction = 1 if pred_prob > 0.5 else 0
    
    return prediction, pred_prob

# Constants
IMG_WIDTH, IMG_HEIGHT = (240, 240)
# Define both possible model file paths - .keras and .model extensions
BEST_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'brain_tumor_model.keras')
ALTERNATIVE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cnn-parameters-improvement-23-0.91.model')

def load_brain_tumor_model():
    """Load the trained brain tumor detection model"""
    # Try loading the .keras model first
    if os.path.exists(BEST_MODEL_PATH):
        return load_model(BEST_MODEL_PATH)
    # Fall back to .model extension
    elif os.path.exists(ALTERNATIVE_MODEL_PATH):
        return load_model(ALTERNATIVE_MODEL_PATH)
    else:
        st.error("Model not found. Please train the model first.")
        st.error(f"Tried paths: {BEST_MODEL_PATH} and {ALTERNATIVE_MODEL_PATH}")
        return None

def process_image_for_prediction(uploaded_file):
    """Process the uploaded image for model prediction"""
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.getvalue())
        
        # Read the image
        image = cv2.imread(tmp_path)
        if image is None:
            st.error("Could not read the uploaded image.")
            return None
        
        # Crop brain contour
        image = crop_brain_contour(image, plot=False)  # Disable plot to avoid matplotlib warnings
        
        # Resize and normalize
        image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        
        # Convert to array and add batch dimension
        image_array = np.array([image])
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return image_array
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles', 'app.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🧠 Brain Tumor Detection</h1>
            <p class="hero-subtitle">AI-Powered MRI Analysis using Deep Learning</p>
            <p class="hero-description">Upload a brain MRI scan to detect the presence of tumors with our advanced CNN model</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_brain_tumor_model()
    
    # Create two columns for the interface
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h3 class="section-header"><span class="section-icon">📤</span> Upload MRI Scan</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...", 
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="📸 Uploaded MRI Image", width="stretch")
            
            # Process image button
            detect_button = st.button("🔍 Analyze MRI Scan", type="primary", use_container_width=True)
            
            if detect_button:
                with st.spinner("🔬 Analyzing MRI scan..."):
                    # Process the image
                    image_array = process_image_for_prediction(uploaded_file)
                    
                    if image_array is not None and model is not None:
                        # Make prediction
                        prediction, probability = predict_tumor(model, image_array)
                        
                        # Show results in the second column
                        with col2:
                            st.markdown('<h3 class="section-header"><span class="section-icon">📊</span> Analysis Results</h3>', unsafe_allow_html=True)
                            
                            # Display processed image
                            try:
                                # Convert from normalized [0,1] float to [0,255] uint8
                                processed_img = (image_array[0] * 255).astype(np.uint8)
                                
                                # OpenCV uses BGR, but Streamlit expects RGB
                                if processed_img.shape[-1] == 3:  # If it has 3 channels (color image)
                                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                
                                st.image(
                                    processed_img,
                                    caption="🧬 Processed MRI Image", 
                                    width="stretch"
                                )
                            except Exception as e:
                                st.warning(f"Could not display processed image: {e}")
                            
                            # Display prediction with modern styling
                            if prediction == 1:
                                st.markdown(f"""
                                    <div class="result-card result-positive">
                                        <div class="result-icon">⚠️</div>
                                        <div class="result-title">Tumor Detected</div>
                                        <div class="result-confidence">Confidence: {probability:.1%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="result-card result-negative">
                                        <div class="result-icon">✅</div>
                                        <div class="result-title">No Tumor Detected</div>
                                        <div class="result-confidence">Confidence: {(1-probability):.1%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Progress bar for confidence
                            confidence_value = float(probability if prediction == 1 else 1-probability)
                            st.markdown('<div class="confidence-section">', unsafe_allow_html=True)
                            st.markdown('<p class="confidence-label">Prediction Confidence</p>', unsafe_allow_html=True)
                            st.progress(confidence_value)
                            st.markdown(f'<p style="color: var(--text-secondary); text-align: center; margin-top: 0.5rem;">{confidence_value:.1%}</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Additional information
                            st.markdown("""
                                <div class="info-card">
                                    <div class="info-icon">ℹ️</div>
                                    <div class="info-content">
                                        <strong>Medical Disclaimer:</strong> This is an AI prediction and should not be considered as a medical diagnosis.
                                        Always consult with a qualified healthcare professional for proper diagnosis and treatment.
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
    
    # If no file is uploaded yet, show some information in the second column
    if not uploaded_file:
        with col2:
            st.markdown('<h3 class="section-header"><span class="section-icon">🤖</span> How It Works</h3>', unsafe_allow_html=True)
            st.markdown("""
                <div class="feature-list">
                    <div class="feature-item">
                        <div class="feature-number">1</div>
                        <div class="feature-content">
                            <strong>Upload Image</strong>
                            <p>Select a brain MRI scan in JPG, JPEG, or PNG format</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-number">2</div>
                        <div class="feature-content">
                            <strong>AI Processing</strong>
                            <p>Our CNN model analyzes the scan, isolating the brain region and normalizing the data</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-number">3</div>
                        <div class="feature-content">
                            <strong>Get Results</strong>
                            <p>Receive instant detection results with confidence scores</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="section-header" style="margin-top: 2rem;"><span class="section-icon">📈</span> Model Performance</h3>', unsafe_allow_html=True)
            
            # Create metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">91%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)
            with metric_col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">240x240</div>
                        <div class="metric-label">Resolution</div>
                    </div>
                """, unsafe_allow_html=True)
            with metric_col3:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">CNN</div>
                        <div class="metric-label">Deep Learning</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="section-header" style="margin-top: 2rem;"><span class="section-icon">🖼️</span> Sample Images</h3>', unsafe_allow_html=True)
            st.markdown('<p class="sample-description">The model was trained on MRI images similar to these:</p>', unsafe_allow_html=True)
            
            # Try to load and display some sample images
            sample_paths = []
            repo_root = os.path.dirname(os.path.abspath(__file__))
            yes_dir = os.path.join(repo_root, 'yes')
            no_dir = os.path.join(repo_root, 'no')
            
            if os.path.exists(yes_dir) and os.path.exists(no_dir):
                # Get one sample from each class
                yes_files = [os.path.join(yes_dir, f) for f in os.listdir(yes_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                no_files = [os.path.join(no_dir, f) for f in os.listdir(no_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if yes_files and no_files:
                    sample_paths = [yes_files[0], no_files[0]]
                    
                    sample_col1, sample_col2 = st.columns(2)
                    
                    with sample_col1:
                        sample_img = cv2.imread(sample_paths[0])
                        if sample_img is not None:
                            st.markdown('<div class="sample-image">', unsafe_allow_html=True)
                            st.image(
                                cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB),
                                caption="✅ Sample: Tumor Present",
                                width="stretch"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                    with sample_col2:
                        sample_img = cv2.imread(sample_paths[1])
                        if sample_img is not None:
                            st.markdown('<div class="sample-image">', unsafe_allow_html=True)
                            st.image(
                                cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB),
                                caption="❌ Sample: No Tumor",
                                width="stretch"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <p>🧠 <strong>Brain Tumor Detection System</strong> • Powered by Deep Learning & TensorFlow</p>
            <p class="footer-disclaimer">⚠️ This is a demo application for educational purposes only. Not intended for actual medical diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()