# PCA + ANN Face Recognition

This project implements a **Face Recognition System** using **Principal Component Analysis (PCA)** for dimensionality reduction and an **Artificial Neural Network (ANN)** (Multi-layer Perceptron) for classification.

It evaluates how the number of eigenfaces (`k`) affects recognition accuracy.

---

## 📂 Project Structure
.
├── pca_ann_face_recognition.py # Main code for PCA + ANN face recognition
├── dataset.zip # Compressed dataset (faces images)
└── README.md # Project documentation

---

## 🚀 Features
- Loads a dataset of face images (grayscale, resized to 50x50).
- Applies **PCA** to compute eigenfaces.
- Projects images into lower-dimensional space.
- Trains an **MLPClassifier** (from scikit-learn).
- Evaluates classification accuracy for different values of `k`.
- Plots **Accuracy vs. Number of Eigenfaces**.

---

## 📦 Requirements
Install the dependencies before running:
```bash
pip install numpy opencv-python scikit-learn matplotlib

🗂️ Dataset

The dataset is provided as dataset.zip.
After cloning the repository, extract it:

unzip dataset.zip -d dataset

Your dataset folder should look like this:

dataset/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        ...


Each subfolder corresponds to one person, containing their face images.

▶️ Usage

Run the program with:

python pca_ann_face_recognition.py

📊 Output

Prints dataset statistics and accuracy for different k values.

Displays a plot of Accuracy vs. Number of Eigenfaces.

Example output:

Loaded 120 images from 6 people.
k=10, Accuracy=0.7500
k=20, Accuracy=0.7916
...

🧠 Concepts Used

PCA (Principal Component Analysis): Reduces dimensionality by computing eigenfaces.

Eigenfaces: The most significant directions of face variation.

ANN (Artificial Neural Network): Classifies projected face signatures.

Evaluation: Accuracy is measured for different numbers of eigenfaces (k).

📌 Notes

Default dataset path in the code is:

DATASET_PATH = "dataset/faces"


Adjust if needed.

Works best if dataset contains multiple images per person.

📜 License

This project is open-source and free to use for learning/research purposes.
