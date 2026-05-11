# 🔢 Handwritten Digit Recogniser

A deep learning project that trains a **Convolutional Neural Network (CNN)** on the MNIST dataset and exposes it through an interactive **Streamlit web GUI** that lets users upload their own handwritten digit images and receive real-time predictions.

---

## 📌 Project Description

This system:

1. **Trains** a CNN model on the MNIST dataset (60,000 training images of digits 0–9).
2. **Evaluates** the model on the MNIST test set (10,000 images), targeting ≥ 99% accuracy.
3. **Serves** the model through a browser-based web interface where users can upload a photo or scan of a handwritten digit and immediately see the predicted class together with a confidence chart for all ten digits.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| Deep Learning Framework | TensorFlow 2 / Keras |
| Dataset | MNIST (via `keras.datasets`) |
| Web GUI | Streamlit |
| Image Processing | Pillow (PIL), NumPy |
| Visualisation | Matplotlib |
| Version Control | Git & GitHub |

---

## 🏗️ CNN Architecture

```
Input (28 × 28 × 1)
│
├─ Conv2D(32) → BatchNorm → ReLU
├─ Conv2D(32) → BatchNorm → ReLU
├─ MaxPooling2D(2×2)
├─ Dropout(0.25)
│
├─ Conv2D(64) → BatchNorm → ReLU
├─ Conv2D(64) → BatchNorm → ReLU
├─ MaxPooling2D(2×2)
├─ Dropout(0.25)
│
├─ Flatten
├─ Dense(128) → BatchNorm → ReLU
├─ Dropout(0.50)
│
└─ Dense(10, softmax)  →  predicted digit (0–9)
```

---

## 📁 Project Structure

```
mnist_digit_recogniser/
├── train_model.py          # CNN training script
├── app.py                  # Streamlit web GUI
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
└── results/                # Generated at runtime
    ├── training_curves.png
    └── evaluation_results.txt
```

---

## 🚀 Instructions to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/swathimegavath/mnist-cnn-digit-classifier.git
cd Machine_Learning_Assignment_Complete
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

> The MNIST dataset (~11 MB) is downloaded automatically on first run.

```bash
python train_model.py
```

Training takes **2–5 minutes** on a modern CPU (faster with GPU).  
On completion you will see:

- `mnist_cnn_model.keras` – the trained model
- `results/training_curves.png` – accuracy & loss plots
- `results/evaluation_results.txt` – final test metrics

### 5. Launch the web GUI

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### 6. Using the GUI

1. Click **Browse files** (or drag-and-drop) to upload a handwritten digit image.
2. Accepted formats: `.png`, `.jpg`, `.jpeg`
3. The app displays:
   - Your uploaded image
   - The predicted digit (0–9) with confidence percentage
   - The preprocessed 28 × 28 version of your image
   - A bar chart showing the model's confidence for all ten digits

>  handwritten digit images from the MNIST dataset. Write a digit on paper and uploaded 

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | ≥ 99.2% |
| Test Loss | ≤ 0.03 |
| Optimizer | Adam (lr = 1e-3) |
| Training epochs | Up to 15 (early stopping) |

---
```

---

## 📎 GitHub Link


https://github.com/swathimegavath/mnist-cnn-digit-classifier

---


