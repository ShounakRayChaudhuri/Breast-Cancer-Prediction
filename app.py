import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the dataset
data = load_breast_cancer()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.Series(data.target)

# Top 15 selected features
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity',
    'mean concave points', 'mean symmetry', 'radius error', 'area error', 'concavity error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst concave points'
]

X = X_full[selected_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Streamlit UI
st.title("🩺 Breast Cancer Predictor (Simplified)")
st.write("Provide inputs to predict if the tumor is **Benign (1)** or **Malignant (0)**.")

# Get min/max for each feature for UI sliders
feature_ranges = {feature: (float(X[feature].min()), float(X[feature].max())) for feature in selected_features}

def get_user_input():
    user_vals = []
    for feature in selected_features:
        min_val, max_val = feature_ranges[feature]
        step = (max_val - min_val) / 100
        value = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=step)
        user_vals.append(value)
    return np.array(user_vals).reshape(1, -1)

user_input = get_user_input()

# Predict
if st.button("Predict"):
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input).max() * 100

    if prediction[0] == 0:
        st.error(f"Prediction: Malignant Tumor ⚠️ (Confidence: {probability:.2f}%)")
    else:
        st.success(f"Prediction: Benign Tumor ✅ (Confidence: {probability:.2f}%)")

# Show accuracy
with st.expander("Model Performance"):
    st.write(f"Training Accuracy: {accuracy_score(Y_train, model.predict(X_train)):.3f}")
    st.write(f"Testing Accuracy: {accuracy_score(Y_test, model.predict(X_test)):.3f}")

# Classification Report
with st.expander("Classification Report"):
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

# Confusion Matrix
with st.expander("Confusion Matrix"):
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], 
                yticklabels=['Malignant', 'Benign'], ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Feature Correlation Heatmap
with st.expander("Feature Correlation Heatmap"):
    correlation_matrix = X.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)