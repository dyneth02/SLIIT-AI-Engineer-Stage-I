import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
print("Step 1: Reading the dataset...")
df = pd.read_csv('Mobile_Price_Classification-220531-204702.txt')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nPrice range distribution:")
print(df['price_range'].value_counts())

# Step 2: Prepare the data
print("\nStep 2: Preparing the data...")

# Separate features and target
X = df.drop('price_range', axis=1)
y = df['price_range']

# For binary classification, we'll convert multi-class to binary
# 0,1 -> 0 (low), 2,3 -> 1 (high)
y_binary = (y >= 2).astype(int)

print(f"\nBinary price distribution:")
print(f"Low (0): {sum(y_binary == 0)}")
print(f"High (1): {sum(y_binary == 1)}")

# Split the data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Build the ANN model
print("\nStep 3: Building the ANN model...")

model = Sequential([
    # Input layer (automatically inferred from input shape)
    Dense(8, activation='relu', input_shape=(X_train_scaled.shape[1],), name='hidden_layer_1'),
    
    # Second hidden layer with 4 neurons
    Dense(4, activation='relu', name='hidden_layer_2'),
    
    # Output layer (binary classification)
    Dense(1, activation='sigmoid', name='output_layer')
])

print("\nModel Architecture:")
model.summary()

# Step 4: Compile the model
print("\nStep 4: Compiling the model...")

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# Step 5: Train the model
print("\nStep 5: Training the model...")
print("Training for 100 epochs with batch size of 32...")

history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Step 6: Save the weights
print("\nStep 6: Saving the model weights...")
model.save_weights('mobile_price_model_weights.h5')
print("Weights saved successfully as 'mobile_price_model_weights.h5'")

# Also save the entire model for easier loading later
model.save('mobile_price_model.h5')
print("Complete model saved as 'mobile_price_model.h5'")

# Save the scaler for future predictions
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")

# Plot training history
print("\nGenerating training history plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy Over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss Over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history plots saved as 'training_history.png'")
plt.show()

# Make some sample predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Test with a few samples from the test set
sample_indices = np.random.choice(len(X_test), 5, replace=False)
samples = X_test_scaled[sample_indices]
predictions = model.predict(samples, verbose=0)
predicted_classes = (predictions > 0.5).astype(int).flatten()
actual_classes = y_test.iloc[sample_indices].values

print("\nSample predictions vs actual:")
for i, (pred_prob, pred_class, actual) in enumerate(zip(predictions, predicted_classes, actual_classes)):
    print(f"Sample {i+1}: Predicted = {pred_class} (prob: {pred_prob[0]:.4f}), Actual = {actual}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nFiles saved:")
print("  - mobile_price_model_weights.h5 (model weights)")
print("  - mobile_price_model.h5 (complete model)")
print("  - scaler.pkl (feature scaler)")
print("  - training_history.png (training plots)")
