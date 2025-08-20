import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Parameters
IMAGE_SIZE = (50, 50)  # Resize all images
DATASET_PATH = "C:/Users/gladson jacob/OneDrive - Amity University/Attachments/Desktop/AI Internship Project/dataset/faces"
K_VALUES = [10, 20, 30, 40, 50, 60, 70]

# 1. Load Dataset
def load_dataset(path):
    images, labels, label_names = [], [], []
    for label_id, person_name in enumerate(sorted(os.listdir(path))):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path):
            continue
        label_names.append(person_name)
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue  # Skip non-image files
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Warning] Could not read image: {img_path}")
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img.flatten())
            labels.append(label_id)
    if not images:
        print("❌ No valid images found.")
        return np.array([]), np.array([]), []
    return np.array(images).T, np.array(labels), label_names

# 2. PCA Functions
def mean_face(Face_Db):
    return np.mean(Face_Db, axis=1).reshape(-1, 1)

def center_faces(Face_Db, M):
    return Face_Db - M

def compute_surrogate_cov(Delta):
    return np.dot(Delta.T, Delta)

def eigen_decomposition(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(-eigvals)  # Descending
    return eigvals[idx], eigvecs[:, idx]

def generate_eigenfaces(Delta, eigvecs, k):
    Psi = eigvecs[:, :k]  # Top-k
    return np.dot(Delta, Psi)  # mn x k

def project_faces(eigenfaces, Delta):
    return np.dot(eigenfaces.T, Delta)

def project_test_face(eigenfaces, test_img):
    return np.dot(eigenfaces.T, test_img)

# 3. Main
def run_pca_ann(k_val, Face_Db, labels):
    M = mean_face(Face_Db)
    Delta = center_faces(Face_Db, M)
    C = compute_surrogate_cov(Delta)
    eigvals, eigvecs = eigen_decomposition(C)
    eigenfaces = generate_eigenfaces(Delta, eigvecs, k_val)
    signatures = project_faces(eigenfaces, Delta)

    X = signatures.T  # shape: (p, k)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 4. Run the evaluation
def main():
    Face_Db, labels, label_names = load_dataset(DATASET_PATH)
    if Face_Db.size == 0:
        print("❌ No images were loaded. Please check the dataset path or image formats.")
        return

    print(f"Loaded {Face_Db.shape[1]} images from {len(label_names)} people.")

    accuracies = []
    for k in K_VALUES:
        acc = run_pca_ann(k, Face_Db, labels)
        print(f"k={k}, Accuracy={acc:.4f}")
        accuracies.append(acc)

    plt.plot(K_VALUES, accuracies, marker='o')
    plt.xlabel("Number of Eigenfaces (k)")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs k in PCA+ANN Face Recognition")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()