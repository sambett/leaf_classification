# 🌱 Plant Disease Classifier

**AI-Powered Plant Health Diagnosis using VGG16 Transfer Learning**

A Streamlit web application for classifying plant diseases with 98% accuracy. Built by **Bettaieb Selma** at the Faculty of Sciences of Sfax.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)

## 🚀 Features

- **98% Accuracy** - VGG16 transfer learning model
- **Real-time Classification** - Upload image and get instant results
- **Professional Interface** - Clean, modern design
- **Treatment Recommendations** - Specific advice for each disease
- **3 Disease Categories**: Healthy, Powdery Mildew, Rust Disease

## 📱 Live Demo

🔗 **[Try the app live on Streamlit Cloud](your-streamlit-url-here)**

## 🏗️ Deployment Guide

### Option 1: Streamlit Cloud (Recommended)

1. **Fork this repository**
2. **Handle the large model file (120MB):**

   **Method A: Git LFS (Large File Storage)**
   ```bash
   # Install Git LFS
   git lfs install
   
   # Track the model file
   git lfs track "*.h5"
   
   # Add and commit
   git add .gitattributes
   git add regularized_vgg16_model.h5
   git commit -m "Add model with Git LFS"
   git push
   ```

   **Method B: External Hosting**
   ```python
   # In app.py, replace model loading with:
   import urllib.request
   
   model_url = "https://your-cloud-storage.com/regularized_vgg16_model.h5"
   if not os.path.exists("regularized_vgg16_model.h5"):
       urllib.request.urlretrieve(model_url, "regularized_vgg16_model.h5")
   ```

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repo
   - Set main file: `app.py`
   - Deploy!

### Option 2: Local Development

```bash
# Clone the repository
git clone your-repo-url
cd streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📦 Repository Structure

```
streamlit/
├── app.py                    # Main Streamlit application
├── regularized_vgg16_model.h5 # Trained VGG16 model (120MB)
├── class_indices.json        # Class mappings
├── requirements.txt          # Python dependencies
├── compress_model.py         # Model compression utility
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔧 Technical Details

### Model Architecture
- **Base Model:** VGG16 (ImageNet pretrained)
- **Strategy:** Partial layer freezing (last 4 layers)
- **Input Size:** 224×224×3
- **Classes:** 3 (Healthy, Powdery Mildew, Rust Disease)
- **Accuracy:** 98% on validation set

### Dependencies
```txt
streamlit>=1.28.1
tensorflow>=2.13.0
numpy>=1.24.3
pillow>=10.0.1
```

## 🧪 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.0% |
| **Model Size** | 120MB |
| **Architecture** | VGG16 Transfer Learning |
| **Training Strategy** | Partial Freezing |

### Per-Class Results
- **Healthy:** 98% F1-Score
- **Powdery Mildew:** 97% F1-Score  
- **Rust Disease:** 99% F1-Score

## 📊 Usage

1. **Upload Image:** Click to upload a plant leaf image
2. **Analyze:** Click "Analyze Plant Health" 
3. **Results:** View predictions with confidence scores
4. **Recommendations:** Get specific treatment advice

## 🎓 Research Background

This project implements the research conducted at **Faculty of Sciences of Sfax** by **Bettaieb Selma**, achieving 98% accuracy using VGG16 transfer learning with partial layer freezing strategy.

### Key Research Insights:
- **Partial freezing** outperformed full training and full freezing
- **Data augmentation** improved model robustness significantly
- **VGG16** performed better than ResNet50 for this specific dataset
- **Transfer learning** proved superior to custom CNN architectures

## 🐛 Troubleshooting

### Large Model File Issues
If you encounter issues with the 120MB model file:

1. **Use Git LFS:** `git lfs track "*.h5"`
2. **External hosting:** Upload to Google Drive/Dropbox
3. **Compression:** Use `python compress_model.py`

### Streamlit Cloud Deployment
- Ensure `requirements.txt` is present
- Model file must be <100MB or use Git LFS
- Python version compatibility (3.8-3.11 recommended)

## 📞 Contact

**Bettaieb Selma**  
Faculty of Sciences of Sfax  
Computer Science Engineering Student  

## 📄 License

This project is for educational and research purposes.

---

*Built with ❤️ using Streamlit and TensorFlow*
