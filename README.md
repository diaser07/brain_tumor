# 🧠 Brain Tumor Classification using CNN (MRI Dataset)

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into **4 tumor types** using the Kaggle dataset:

📌 **Dataset Used:**  
Masoud Nickparvar — *Brain Tumor MRI Dataset*  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The model is trained on four classes:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

A Streamlit web app is included to allow users to upload MRI scans and receive predictions.

---

## 📌 Features

- ✔️ Trained on a 4-class Brain Tumor MRI dataset  
- ✔️ Deep Learning model (TensorFlow/Keras)  
- ✔️ Data augmentation for better generalization  
- ✔️ Clean Streamlit web app for predictions  
- ✔️ GPU-optimized training (works in Google Colab)  
- ✔️ High accuracy with CNN / transfer learning  

---

## 📂 Project Structure

Brain-Tumor-Classification/
│── model/
│ ├── brain_tumor_model.h5
│── dataset/ (from Kaggle)
│ ├── Training/
│ │ ├── glioma/
│ │ ├── meningioma/
│ │ ├── pituitary/
│ │ ├── no_tumor/
│ ├── Testing/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ ├── no_tumor/
│── app.py
│── train.py
│── requirements.txt
│── README.md

---

## 🚀 How to Run Locally

### **1️⃣ Clone the repository**
```bash
git clone https://https://github.com/diaser07/brain_tumor.git
cd Brain-Tumor-Classification

pip install -r requirements.txt
3️⃣ Train the model (Optional — model.h5 already included)

python train.py

4️⃣ Run the Streamlit app

streamlit run app.py

🧪 Dataset Information

The dataset contains MRI images divided into:

Class	      Description
Glioma	    Brain tumor that affects glial cells

Meningioma	Tumor in meninges (outer brain layer)

Pituitary	  Tumor located near pituitary gland

No Tumor	  Healthy brain MRI

Image sources include multiple MRI machines and orientations.

🧠 Model Architecture

The training notebook uses:

CNN (Convolutional Neural Network)

Conv2D + MaxPooling layers

Batch Normalization

Dropout (to prevent overfitting)

Dense layers + Softmax

Loss Function: categorical_crossentropy
Optimizer: Adam
Metrics: accuracy

Accuracy usually reaches 85–95% depending on epochs and augmentation.

🌐 Streamlit Web App

The app allows:

Uploading MRI images

Auto-resizing + normalization

Predicting tumor type

Showing confidence scores

🖼️ Screenshots



<img width="1747" height="742" alt="example" src="https://github.com/user-attachments/assets/02708b31-35ea-4594-9b74-57b4ece9ca9e" />

