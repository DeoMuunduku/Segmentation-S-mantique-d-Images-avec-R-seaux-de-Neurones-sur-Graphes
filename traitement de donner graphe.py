import os
import numpy as np
from PIL import Image

# Fonction pour charger les images et les annotations
def load_data(image_dir, label_dir):
    image_filenames = os.listdir(image_dir)
    label_filenames = os.listdir(label_dir)
    images = []
    labels = []
    for image_filename in image_filenames:
        if image_filename.endswith('.png'):
            image = np.array(Image.open(os.path.join(image_dir, image_filename)))
            images.append(image)
    for label_filename in label_filenames:
        if label_filename.endswith('.png'):
            label = np.array(Image.open(os.path.join(label_dir, label_filename)))
            labels.append(label)
    return np.array(images), np.array(labels)

# Fonction pour prétraiter les données
def preprocess_data(images, labels, input_shape, num_classes):
    # Redimensionner les images et les annotations à la forme spécifiée par input_shape
    resized_images = []
    resized_labels = []
    for image, label in zip(images, labels):
        resized_image = np.array(Image.fromarray(image).resize(input_shape[:2]))
        resized_label = np.array(Image.fromarray(label).resize(input_shape[:2], resample=Image.NEAREST))
        resized_labels.append(resized_label)
        resized_images.append(resized_image)
    # Normaliser les images (valeur entre 0 et 1)
    normalized_images = np.array(resized_images) / 255.0
    # Adapter les annotations à un format one-hot encoding
    one_hot_labels = tf.one_hot(resized_labels, depth=num_classes)
    return normalized_images, one_hot_labels

# Définir vos répertoires d'images et d'annotations
train_image_dir = '/home/deo/Images/image'
train_label_dir = '/home/deo/Images/image_labels'
val_image_dir = '/home/deo/Images/image'
val_label_dir = '_/home/deo/Images/imagelabels'

# Charger les données d'entraînement et de validation
train_images, train_labels = load_data(train_image_dir, train_label_dir)
val_images, val_labels = load_data(val_image_dir, val_label_dir)

# Paramètres
input_shape = (256, 256, 3)  # Définir la forme de vos images d'entrée
num_classes = 2               # Nombre de classes (par exemple, 2 pour la segmentation binaire)

# Prétraiter les données
train_images_preprocessed, train_labels_preprocessed = preprocess_data(train_images, train_labels, input_shape, num_classes)
val_images_preprocessed, val_labels_preprocessed = preprocess_data(val_images, val_labels, input_shape, num_classes)

