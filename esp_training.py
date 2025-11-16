import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# ================================================
# 1. Load dataset (64 amp + 64 phase + label)
# ================================================
print("📂 Loading dataset...")
data = pd.read_csv("indexed_csi.csv")

print("Dataset shape:", data.shape)
print("Label distribution:\n", data["label"].value_counts())

# ================================================
# 2. Feature columns
# ================================================
amp_cols = [f"amp_{i}" for i in range(64)]
phase_cols = [f"phase_{i}" for i in range(64)]
feature_cols = amp_cols + phase_cols  # total 128 features

X = data[feature_cols].values
y = data["label"].values

# ================================================
# 3. Encode labels
# ================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel Mapping:")
for cls, enc in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{cls} → {enc}")

# ================================================
# 4. Scale features
# ================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================
# 5. Train-test split
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ================================================
# 6. Train Random Forest
# ================================================
print("\n🌲 Training Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=40,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ================================================
# 7. Evaluate model
# ================================================
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🔵 Test Accuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ================================================
# 8. Save Model + Scaler + Encoder
# ================================================
os.makedirs("esp_model", exist_ok=True)

joblib.dump(rf, "esp_model/rf_model.pkl")
joblib.dump(scaler, "esp_model/scaler.pkl")
joblib.dump(label_encoder, "esp_model/labels.pkl")

print("\n💾 Saved:")
print(" - esp_model/rf_model.pkl")
print(" - esp_model/scaler.pkl")
print(" - esp_model/labels.pkl")
print("\n🎉 Training completed!")
