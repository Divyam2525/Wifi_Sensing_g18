# ==============================
#  Random Forest Classification for CSI Dataset
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ------------------------------
# 1. Load dataset
# ------------------------------
print("📂 Loading dataset...")
data = pd.read_csv("raw_data.csv")
print(f"✅ Dataset loaded: {data.shape[0]} samples, {data.shape[1]} columns")

# Select amplitude and phase features
amp_cols = [col for col in data.columns if col.startswith('amp_')]
phase_cols = [col for col in data.columns if col.startswith('phase_')]
feature_cols = amp_cols + phase_cols

X = data[feature_cols]
y = data['type']   # target: organic / metallic

# ------------------------------
# 2. Encode target labels
# ------------------------------
print("🎯 Encoding target labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)   # organic→0, metallic→1

# ------------------------------
# 3. Normalize features
# ------------------------------
print("⚙️  Scaling feature values...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. Split into train/test sets
# ------------------------------
print("✂️  Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.05, random_state=42, stratify=y_encoded
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ------------------------------
# 5. Train Random Forest model
# ------------------------------
print("🌲 Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=400,      # number of trees
    max_depth=35,          # controls overfitting
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("✅ Model training complete.")

# ------------------------------
# 6. Evaluate model
# ------------------------------
print("\n📊 Evaluating model...")
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {round(acc * 100, 2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ------------------------------
# 7. Confusion matrix visualization
# ------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Organic vs Metallic")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ------------------------------
# 8. Feature importance
# ------------------------------
print("\n🔍 Plotting top feature importances...")
importances = pd.Series(rf.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,5))
sns.barplot(x=top_features.index, y=top_features.values)
plt.title("Top 15 Most Important CSI Features")
plt.xlabel("Feature Name")
plt.ylabel("Importance Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ------------------------------
# 9. Save trained model and preprocessing tools
# ------------------------------
print("\n💾 Saving model and preprocessing tools...")
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/random_forest_csi_model.pkl")
joblib.dump(scaler, "models/csi_scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("🎉 Model, scaler, and label encoder saved successfully in the 'models/' folder!")

# ------------------------------
# 10. Summary
# ------------------------------
print("\n✅ Training complete!")
print("Use the saved model later with joblib.load('models/random_forest_csi_model.pkl') to predict new CSI data.")
