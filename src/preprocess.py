import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

print("📊 Phase 2: Preprocessing & Feature Engineering")
print("=" * 60)

# Load combined dataset
print("\n📂 Loading dataset_combined.csv...")
df = pd.read_csv('dataset_combined.csv')

print(f"✅ Loaded {len(df)} samples")
print(f"📋 Classes: {sorted(df['label'].unique())}")

# Extract labels and features
y = df['label'].values
X = df.drop('label', axis=1).values

print(f"\n🔧 Preprocessing...")
print(f"   Original shape: {X.shape}")

# Normalize: subtract wrist point (first 3 values: x0, y0, z0)
X_normalized = X.copy()
for i in range(len(X)):
    wrist = X[i][:3]  # x0, y0, z0
    # Reshape to 21 landmarks of 3 coords each
    landmarks = X[i].reshape(21, 3)
    landmarks = landmarks - wrist  # Position-invariant
    X_normalized[i] = landmarks.flatten()

print(f"   ✅ Normalized (subtracted wrist point)")

# Scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_normalized)
print(f"   ✅ Scaled to [0, 1]")

# Create processed folder
os.makedirs('processed', exist_ok=True)

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n📊 Data split:")
print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
print(f"   Val:   {len(X_val)} samples ({len(X_val)/len(X_scaled)*100:.1f}%)")
print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")

# Save as numpy arrays
np.save('processed/X_train.npy', X_train)
np.save('processed/X_val.npy', X_val)
np.save('processed/X_test.npy', X_test)
np.save('processed/y_train.npy', y_train)
np.save('processed/y_val.npy', y_val)
np.save('processed/y_test.npy', y_test)

print(f"\n✅ Saved numpy arrays to processed/")

# Save scaler
joblib.dump(scaler, 'processed/scaler.pkl')
print(f"✅ Saved scaler.pkl")

# Save class mapping
class_labels = sorted(df['label'].unique())
class_mapping = {i: label for i, label in enumerate(class_labels)}
with open('processed/class_mapping.json', 'w') as f:
    json.dump(class_mapping, f)
print(f"✅ Saved class_mapping.json")

print(f"\n{'='*60}")
print(f"🎉 Preprocessing complete!")
print(f"   Ready for Phase 3: Model Training")