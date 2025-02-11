import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

IMAGE_REDUCED_SIZE = 64
FOLDER = '../dataset/'  

def showImage(image):
    print("Shape:", image.shape)
    plt.imshow(image)
    plt.show()

def load_data(folder, max_samples=None):
    filenames = []
    labels = []
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg')):
            filenames.append(os.path.join(folder, filename))
            labels.append(filename.split('_')[0])
    
    dataInputs = []
    for i, filename in enumerate(filenames):
        if max_samples is not None and i >= max_samples:
            break
        try:
            image = skimage.io.imread(filename)
  
            image_resized = skimage.transform.resize(image, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE), anti_aliasing=True)
            image_flat = image_resized.reshape(-1)
            dataInputs.append(image_flat)
        except Exception as e:
            print(f"Erreur lors du chargement de {filename} : {e}")
    
    dataInputs = np.array(dataInputs)
    labels = np.array(labels[:len(dataInputs)]) 
    return dataInputs, labels

if __name__ == "__main__":
    print("Chargement des données depuis :", FOLDER)
    X, y = load_data(FOLDER, max_samples=30)  
    print("Nombre d'échantillons chargés :", X.shape[0])
    print("Dimension de chaque échantillon :", X.shape[1])
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    mlp = MLPClassifier(hidden_layer_sizes=(128,),
                        activation='relu',
                        solver='adam',
                        max_iter=100,
                        random_state=42,
                        verbose=True)
    
    print("Entraînement du modèle...")
    mlp.fit(X_train, y_train)
    
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(6, 4))
        plt.plot(mlp.loss_curve_, label='Loss')
        plt.title("Courbe d'apprentissage (Loss)")
        plt.xlabel("Itérations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("Aucune courbe d'apprentissage disponible.")
    
    y_train_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("Accuracy sur le jeu d'entraînement : {:.2f}%".format(train_acc * 100))
    
    y_test_pred = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("Accuracy sur le jeu de test : {:.2f}%".format(test_acc * 100))
