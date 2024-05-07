import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
labels = pd.read_csv('labels.csv')
dataset = pd.read_csv('dataset.csv')
dataset.drop(['Unnamed: 0', 'En.Anterior.', 'idEye'], axis=1, inplace=True)
labels.drop(['Unnamed: 0', 'Data.PLOS_One.idEye'], axis=1, inplace=True)

# Apply label encoding
dataset = dataset.apply(LabelEncoder().fit_transform)
labels = labels.apply(LabelEncoder().fit_transform)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print('\nTest set accuracy: {0:0.2f}%\n'.format(acc))

# Save the model
model.save('D:\\Uni\\Advance AI and Health Care\\Keratoconus-detection\\my_model.keras')

# Load the model
loaded_model = tf.keras.models.load_model('D:\\Uni\\Advance AI and Health Care\\Keratoconus-detection\\my_model.keras')

# Test the loaded model
loss, acc = loaded_model.evaluate(X_test, y_test)
print('\nTest set accuracy with loaded model: {0:0.2f}%\n'.format(acc))
