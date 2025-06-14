import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import time

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .status-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .healthy { border-left: 4px solid #28a745; }
    .powdery { border-left: 4px solid #ffc107; }
    .rust { border-left: 4px solid #dc3545; }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
    }
    
    .refresh-btn {
        position: absolute;
        top: 15px;
        right: 20px;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Model loading with better error handling
@st.cache_resource
def load_model_and_setup():
    """Load model and setup classification"""
    
    # Class configuration
    class_names = ['Healthy', 'Powdery Mildew', 'Rust Disease']
    
    # Create/load class indices
    class_indices = {"Healthy": 0, "Powdery": 1, "Rust": 2}
    if not os.path.exists("class_indices.json"):
        with open("class_indices.json", 'w') as f:
            json.dump(class_indices, f)
    
    model = None
    model_status = "Loading..."
    
    try:
        import tensorflow as tf
        
        # Clear any existing sessions
        tf.keras.backend.clear_session()
        
        # GPU setup if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Try to load the model
        model_path = "regularized_vgg16_model.h5"
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                model_status = f"✅ Model loaded successfully (TensorFlow {tf.__version__})"
            except Exception as e:
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model_status = f"✅ Model loaded (no compile) - TensorFlow {tf.__version__}"
                except Exception as e2:
                    model_status = f"❌ Model loading failed: {str(e2)}"
        else:
            model_status = "❌ Model file not found: regularized_vgg16_model.h5"
            
    except ImportError:
        model_status = "❌ TensorFlow not available - Install TensorFlow first"
    except Exception as e:
        model_status = f"❌ Setup error: {str(e)}"
    
    return model, class_names, model_status

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, image, class_names):
    """Make prediction using the model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get results
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])  # Convert float32 to float
        predicted_class = class_names[predicted_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': [float(p) for p in predictions],  # Convert all to float
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def get_treatment_recommendations(disease_class):
    """Get treatment recommendations based on disease"""
    recommendations = {
        'Healthy': [
            "✅ Continue current care routine",
            "🔍 Monitor plant regularly for any changes",
            "💧 Maintain proper watering schedule",
            "🌱 Ensure good growing conditions"
        ],
        'Powdery Mildew': [
            "🚨 Apply fungicide treatment immediately",
            "💨 Improve air circulation around plant",
            "✂️ Remove and dispose of affected leaves",
            "💧 Avoid overhead watering",
            "🌡️ Reduce humidity levels"
        ],
        'Rust Disease': [
            "🔴 URGENT: Isolate plant from others",
            "🧪 Apply copper-based fungicide",
            "🗑️ Remove and destroy infected material",
            "💨 Improve drainage and air flow",
            "👀 Monitor nearby plants for spread"
        ]
    }
    return recommendations.get(disease_class, recommendations['Healthy'])

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Plant Disease Classifier</h1>
        <p>AI-Powered Plant Health Diagnosis</p>
        <small>Research by Bettaieb Selma - Faculty of Sciences of Sfax</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and setup
    model, class_names, model_status = load_model_and_setup()
    
    # Status display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if "✅" in model_status:
            st.markdown(f'<div class="status-success">{model_status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-error">{model_status}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Model"):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Main interface
    if model is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Upload Plant Image")
            
            uploaded_file = st.file_uploader(
                "Choose a plant leaf image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
                help="Upload a clear image of a plant leaf for disease analysis"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Image info
                file_size = len(uploaded_file.getvalue()) / 1024
                st.info(f"📊 Image: {image.size[0]}×{image.size[1]}px • {file_size:.1f}KB • {image.mode} mode")
                
                # Analyze button
                if st.button("🔍 Analyze Plant Health", type="primary"):
                    with st.spinner("🧠 AI is analyzing your plant..."):
                        time.sleep(1)  # Show spinner for user experience
                        
                        # Make prediction
                        result = predict_disease(model, image, class_names)
                        
                        if result['success']:
                            st.session_state.analysis_result = result
                            st.success("✅ Analysis complete!")
                        else:
                            st.error(f"❌ Analysis failed: {result['error']}")
        
        with col2:
            st.subheader("📊 Results")
            
            if hasattr(st.session_state, 'analysis_result') and uploaded_file is not None:
                result = st.session_state.analysis_result
                
                # Main prediction display
                disease_class = result['class']
                confidence = result['confidence']
                
                # Styling based on disease type
                if disease_class == 'Healthy':
                    card_class = "healthy"
                    status_emoji = "✅"
                elif disease_class == 'Powdery Mildew':
                    card_class = "powdery"  
                    status_emoji = "⚠️"
                else:  # Rust Disease
                    card_class = "rust"
                    status_emoji = "🚨"
                
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h3>{status_emoji} {disease_class}</h3>
                    <h4>{confidence:.1%} Confidence</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                
                # All predictions
                st.subheader("📈 All Predictions")
                predictions = result['all_predictions']
                
                for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
                    col_name, col_bar, col_val = st.columns([2, 3, 1])
                    
                    with col_name:
                        if i == np.argmax(predictions):
                            st.markdown(f"**{class_name}** ⭐")
                        else:
                            st.markdown(class_name)
                    
                    with col_bar:
                        st.progress(prob)
                    
                    with col_val:
                        st.markdown(f"{prob:.1%}")
                
                # Treatment recommendations
                st.subheader("💡 Recommendations")
                recommendations = get_treatment_recommendations(disease_class)
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
                # Technical details in expander
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
                    **Model:** VGG16 Transfer Learning  
                    **Accuracy:** 98% (Research validated)  
                    **Classes:** {len(class_names)} disease categories  
                    **Input Size:** 224x224 pixels  
                    **Preprocessing:** RGB normalization  
                    
                    **Raw Predictions:**
                    """)
                    
                    for class_name, prob in zip(class_names, predictions):
                        st.text(f"{class_name}: {prob:.6f}")
            
            else:
                st.info("👆 Upload an image and click 'Analyze' to see results")
                
                st.markdown("""
                ### 🎯 Disease Categories
                
                **🟢 Healthy**  
                Normal, healthy plant tissue
                
                **🟡 Powdery Mildew**  
                White, powdery fungal coating
                
                **🔴 Rust Disease**  
                Orange-brown pustules/spots
                """)
    
    else:
        st.error("🚫 Model not available. Please check the model file and TensorFlow installation.")
        
        st.markdown("""
        ### 🔧 Troubleshooting Steps:
        
        1. **Check model file:** Ensure `regularized_vgg16_model.h5` exists
        2. **Install TensorFlow:** `pip install tensorflow==2.13.0`
        3. **Check Python version:** TensorFlow requires Python 3.8-3.11
        4. **Click Refresh:** Try the refresh button above
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>
        🎓 Research Project - Faculty of Sciences of Sfax<br>
        👥 Bettaieb Selma<br>
        🧠 VGG16 Transfer Learning Model (98% Accuracy)
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
