import numpy as np
import joblib
import json
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🤖 Phase 3: Model Training & Evaluation")
print("=" * 60)

# Load preprocessed data
print("\n📂 Loading preprocessed data...")
X_train = np.load('processed/X_train.npy')
X_val = np.load('processed/X_val.npy')
X_test = np.load('processed/X_test.npy')
y_train = np.load('processed/y_train.npy', allow_pickle=True)
y_val = np.load('processed/y_val.npy', allow_pickle=True)
y_test = np.load('processed/y_test.npy', allow_pickle=True)

print(f"✅ Loaded training data: {X_train.shape}")

# Load class mapping
with open('processed/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)
class_names = list(class_mapping.values())

print(f"📋 Classes: {class_names}")

# Train SVM model
print("\n🔧 Training SVM model (RBF kernel)...")
model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
model.fit(X_train, y_train)
print("✅ SVM trained")

# Evaluate on validation set
print("\n📊 Evaluating on validation set...")
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"   Validation Accuracy: {val_accuracy*100:.2f}%")

if val_accuracy < 0.90:
    print(f"\n⚠️  Accuracy is below 90%. Consider tuning hyperparameters.")
    print("   Hyperparameter suggestions:")
    print("   - Increase C (e.g., C=50, C=100)")
    print("   - Try different gamma values")
    print("   - Collect more data (currently only 30 samples per sign)")

# Evaluate on test set
print("\n📊 Evaluating on test set...")
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names))

# Create models folder
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/svm_model.pkl')
print("\n✅ Saved model to models/svm_model.pkl")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - SVM Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=100)
print("✅ Saved confusion matrix to assets/confusion_matrix.png")

# Save model info
model_info = {
    'model_type': 'SVM',
    'kernel': 'rbf',
    'C': 10,
    'val_accuracy': float(val_accuracy),
    'test_accuracy': float(test_accuracy),
    'num_classes': len(class_names),
    'input_features': 63,
    'classes': class_names
}
with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print("✅ Saved model_info.json")

print(f"\n{'='*60}")
print(f"🎉 Model Training Complete!")
print(f"   Val Accuracy: {val_accuracy*100:.2f}%")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Ready for Phase 4: Real-Time Detection")