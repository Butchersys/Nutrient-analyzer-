import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import pickle
import shutil

# Configuration section
DATASET_PATH = r'C:\path\to\your\dataset'  # Replace with your dataset path
CLASSIFICATION_MODEL_PATH = r'C:\path\to\your\classification_model'  # Replace with your path
REGRESSION_MODEL_PATH = r'C:\path\to\your\regression_model'  # Replace with your path
INDEX_TO_CLASS_PATH = r'C:\path\to\your\index_to_class.pkl'  # Replace with your path
ZIPPED_MODELS_PATH = r'C:\path\to\your\zipped_models'  # Replace with your path
EXAMPLE_IMAGE_PATH = r'C:\path\to\example\image.jpg'  # Replace with your image path

# Load images, labels, and weights into a DataFrame
labels = os.listdir(DATASET_PATH)
df = pd.DataFrame(columns=['img_path', 'label', 'weight'])

for label in labels:
    img_dir_path = os.path.join(DATASET_PATH, label)
    for img in os.listdir(img_dir_path):
        img_path = os.path.join(img_dir_path, img)
        weight = float(img.split('_')[0])  # Assuming weight is the prefix in the filename
        df.loc[len(df)] = [img_path, label, weight]

# Shuffle and split the dataset
df = df.sample(frac=1).reset_index(drop=True)
x_train, x_val, y_train, y_val = train_test_split(df[['img_path', 'label']], df['weight'], random_state=42, test_size=0.2)

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Classification Data Generator
train_classification_dataset = datagen.flow_from_dataframe(
    dataframe=x_train,
    x_col='img_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_classification_dataset = datagen.flow_from_dataframe(
    dataframe=x_val,
    x_col='img_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Regression Data Generator
train_regression_dataset = datagen.flow_from_dataframe(
    dataframe=x_train,
    x_col='img_path',
    y_col='weight',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Regression task
    shuffle=True,
    seed=42
)

val_regression_dataset = datagen.flow_from_dataframe(
    dataframe=x_val,
    x_col='img_path',
    y_col='weight',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Regression task
    shuffle=False
)

# Build and compile classification model
def build_classification_model(base_model):
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(len(class_indices), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and compile regression model
def build_regression_model(base_model):
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        BatchNormalization(),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create class indices for classification
class_indices = {label: idx for idx, label in enumerate(df['label'].unique())}
index_to_class = {v: k for k, v in class_indices.items()}

# Create and train classification model
base_model_classification = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
classification_model = build_classification_model(base_model_classification)
classification_model.fit(
    train_classification_dataset,
    epochs=10,
    steps_per_epoch=train_classification_dataset.samples // 32
)

# Create and train regression model
base_model_regression = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
regression_model = build_regression_model(base_model_regression)
regression_model.fit(
    train_regression_dataset,
    epochs=10,
    steps_per_epoch=train_regression_dataset.samples // 32
)

# Evaluate the regression model
def evaluate_regression_model(model, val_dataset):
    val_labels = []
    predictions = []
    for _ in range(val_dataset.samples // val_dataset.batch_size):
        val_images, batch_labels = next(val_dataset)
        preds = model.predict(val_images)
        predictions.extend(preds.flatten())
        val_labels.extend(batch_labels)
    mse = mean_squared_error(val_labels, predictions)
    return mse

# Evaluate the classification model
def evaluate_classification_model(model, val_dataset):
    all_labels = []
    all_predictions = []
    for _ in range(val_dataset.samples // val_dataset.batch_size):
        val_images, val_labels = next(val_dataset)
        predictions = model.predict(val_images)
        predicted_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_classes)
        all_labels.extend(np.argmax(val_labels, axis=1))
    
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=list(class_indices.keys()))
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return accuracy, report, conf_matrix

mse_regression = evaluate_regression_model(regression_model, val_regression_dataset)
print(f"Validation Mean Squared Error (Regression): {mse_regression}")

accuracy_classification, class_report, conf_matrix = evaluate_classification_model(classification_model, val_classification_dataset)
print(f"Validation Accuracy (Classification): {accuracy_classification}")
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)

# Save the models and class mappings
classification_model.save(CLASSIFICATION_MODEL_PATH)
regression_model.save(REGRESSION_MODEL_PATH)

with open(INDEX_TO_CLASS_PATH, 'wb') as f:
    pickle.dump(index_to_class, f)

# Zip the saved models
shutil.make_archive(ZIPPED_MODELS_PATH, 'zip', os.path.dirname(CLASSIFICATION_MODEL_PATH))

# Example of loading models and making predictions
loaded_classification_model = load_model(CLASSIFICATION_MODEL_PATH)
loaded_regression_model = load_model(REGRESSION_MODEL_PATH)

with open(INDEX_TO_CLASS_PATH, 'rb') as f:
    index_to_class = pickle.load(f)

# Example image prediction
img = load_img(EXAMPLE_IMAGE_PATH, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict weight
predicted_weight = loaded_regression_model.predict(img_array)
print(f"Predicted weight: {predicted_weight[0][0]}")

# Predict food name
predictions = loaded_classification_model.predict(img_array)
predicted_class_index = np.argmax(predictions)
predicted_class_name = index_to_class[predicted_class_index]

print(f"Predicted food name: {predicted_class_name}")
