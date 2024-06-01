import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Load model and data
model = pickle.load(open('random_forest_model.pkl', 'rb'))
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
X_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').values.ravel()

# Streamlit app
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Model Performance", "Predict Diabetes"])

if menu == "Home":
    st.title("Diabetes Prediction Dashboard")
    st.write("Selamat datang di dashboard prediksi diabetes. Anda dapat menjelajahi performa model dan melakukan prediksi menggunakan menu di sidebar.")

if menu == "Model Performance":
    st.title("Model Performance")

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)

    folds = 10
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
    mean_cv_score = np.mean(cv_scores)

    st.write("### Training Metrics")
    st.write(f"Accuracy: {train_accuracy:.2f}")
    st.write(f"Precision: {train_precision:.2f}")
    st.write(f"Recall: {train_recall:.2f}")
    st.write(f"F1-score: {train_f1:.2f}")
    st.write(f"ROC-AUC Score: {train_roc_auc:.2f}")

    st.write("### Validation Metrics")
    st.write(f"Accuracy: {val_accuracy:.2f}")
    st.write(f"Precision: {val_precision:.2f}")
    st.write(f"Recall: {val_recall:.2f}")
    st.write(f"F1-score: {val_f1:.2f}")
    st.write(f"ROC-AUC Score: {val_roc_auc:.2f}")
    st.write(f"Mean Cross Validation Score: {mean_cv_score:.2f}")

    st.write("### Confusion Matrix")
    st.write("#### Training Confusion Matrix")
    st.write(train_conf_matrix)

    st.write("#### Validation Confusion Matrix")
    st.write(val_conf_matrix)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].matshow(train_conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(train_conf_matrix.shape[0]):
        for j in range(train_conf_matrix.shape[1]):
            ax[0].text(x=j, y=i, s=train_conf_matrix[i, j], va='center', ha='center')
    ax[0].set_title('Training Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    ax[1].matshow(val_conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(val_conf_matrix.shape[0]):
        for j in range(val_conf_matrix.shape[1]):
            ax[1].text(x=j, y=i, s=val_conf_matrix[i, j], va='center', ha='center')
    ax[1].set_title('Validation Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('True')

    st.pyplot(fig)

if menu == "Predict Diabetes":
    st.title("Predict Diabetes")
    st.sidebar.header("Input Data Pasien")

    def user_input_features():
        glucose = st.sidebar.slider('Glucose', 0, 200, 117)
        blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
        skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
        insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
        bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
        diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.3725)
        age = st.sidebar.slider('Age', 21, 81, 29)
        
        data = {
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree_function,
                'Age': age}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader('Input Data')
    st.write(input_df)

    if st.button('Predict'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Hasil Prediksi')
        diabetes_labels = np.array(['Non-Diabetes', 'Diabetes'])
        st.write(diabetes_labels[prediction])

        st.subheader('Probabilitas Prediksi')
        st.write(prediction_proba)
