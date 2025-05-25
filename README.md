# 🌞 Skin Cancer Detection using Deep Learning

A deep learning-based solution to detect different types of skin cancer from dermatoscopic images. This project leverages Convolutional Neural Networks (CNNs) for image classification and provides a fast, reliable, and cost-effective method for early skin cancer detection.

## 🧠 Features

* Detects skin lesions using image classification
* Trained on HAM10000 dermatology dataset
* Achieves high accuracy using CNN / transfer learning (e.g., ResNet50, EfficientNet)
* Visualizes predictions with confidence scores
* Web interface / Android app (optional frontend)

## 📁 Dataset

* **Name**: HAM10000 ("Human Against Machine with 10000 training images")
* **Source**: [Kaggle Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
* **Classes**:

  * Melanocytic nevi (nv)
  * Melanoma (mel)
  * Benign keratosis-like lesions (bkl)
  * Basal cell carcinoma (bcc)
  * Actinic keratoses (akiec)
  * Vascular lesions (vasc)
  * Dermatofibroma (df)

## 🏗️ Tech Stack

* **Language**: Python
* **Libraries**: TensorFlow / Keras, OpenCV, NumPy, Matplotlib, Scikit-learn
* **Optional**: Streamlit / Flask for web UI | Kotlin (Jetpack Compose) for Android

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the model training
python train.py

# Predict new images
python predict.py --image path_to_image.jpg
```

## 📊 Results

* Accuracy: \~87% on test set
* Confusion matrix and ROC curves included in `/results` folder

## 🧪 Future Improvements

* Real-time detection from camera feed
* Improve model with ensemble methods
* Build mobile or web app for wider accessibility

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

