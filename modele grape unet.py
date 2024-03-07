import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU

# Définir l'architecture du modèle U-Net
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # Encodeur
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Décodeur
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    up1 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(256, 2, activation='relu', padding='same')(up1)
    merge1 = concatenate([conv3, conv5], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D(128, 2, activation='relu', padding='same')(up2)
    merge2 = concatenate([conv2, conv7], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    up3 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D(64, 2, activation='relu', padding='same')(up3)
    merge3 = concatenate([conv1, conv9], axis=3)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv10 = Conv2D(64, 3, activation='relu', padding='same')(conv10)
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv10)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Définir vos données d'entraînement et de validation
train_images = /home/deo/Images/image # Chargement de vos images d'entraînement
train_labels = /home/deo/Images/image_labels # Chargement de vos annotations pour les images d'entraînement
val_images = /home/deo/Images/image_test   # Chargement de vos images de validation
val_labels = /home/deo/Images/image_test_anota   # Chargement de vos annotations pour les images de validation

# Paramètres
input_shape = (height, width, num_channels)  # Définir la forme de vos images
num_classes = num_classes                    # Nombre de classes (par exemple, 2 pour la segmentation binaire)
num_epochs = 10                               # Nombre d'époques
batch_size = 16                               # Taille du batch

# Construire le modèle
model = build_model(input_shape, num_classes)

# Compiler le modèle
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[MeanIoU(num_classes)])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

# Évaluer le modèle
evaluation = model.evaluate(val_images, val_labels)
print("Loss:", evaluation[0])
print("Mean IoU:", evaluation[1])
.............suite du code .....
