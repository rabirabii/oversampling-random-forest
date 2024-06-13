import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import KFold, cross_val_score
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import sklearn
print(sklearn.__version__)
# Load model and data
model = pickle.load(open('random_forest_model.pkl', 'rb'))
x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
x_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv").values.ravel()

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Model Performance", "Predict Diabetes"])

# Homepage
if menu == "Home":
    st.title("Diabetes Prediction Dashboard")
    st.write("Welcome to the Diabetes Prediction Dashboard. Use the sidebar to navigate to different sections.")

if menu == "Model Performance":
    st.title("Model Performance")
    
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    train_precision = precision_score(y_train, y_train_pred) * 100
    train_recall = recall_score(y_train, y_train_pred) * 100
    train_f1 = f1_score(y_train, y_train_pred) * 100
    train_roc_auc = roc_auc_score(y_train, y_train_pred) * 100
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    
    val_accuracy = accuracy_score(y_val, y_val_pred) * 100
    val_precision = precision_score(y_val, y_val_pred) * 100
    val_recall = recall_score(y_val, y_val_pred) * 100
    val_f1 = f1_score(y_val, y_val_pred) * 100
    val_roc_auc = roc_auc_score(y_val, y_val_pred) * 100
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    
    folds = 10
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x_train, y_train, cv=kf)
    mean_cv_score = np.mean(cv_scores) * 100
    
    st.write("### Training Metrics")
    st.write(f"Accuracy: {train_accuracy:.2f}%")
    st.write(f"Precision: {train_precision:.2f}%")
    st.write(f"Recall: {train_recall:.2f}%")
    st.write(f"F1-Score: {train_f1:.2f}%")
    st.write(f"ROC-AUC: {train_roc_auc:.2f}%")
  
    st.write("### Validation Metrics")
    st.write(f"Accuracy: {val_accuracy:.2f}%")
    st.write(f"Precision: {val_precision:.2f}%")
    st.write(f"Recall: {val_recall:.2f}%")
    st.write(f"F1-Score: {val_f1:.2f}%")
    st.write(f"ROC-AUC: {val_roc_auc:.2f}%")
    
    st.write("#### Mean Cross Validation")
    st.write(f"Mean Cross Validation: {mean_cv_score:.2f}%")
    
    st.write("### Confusion Matrix")
    st.write("#### Training Confusion Matrix")
    train_conf_matrix_fig = ff.create_annotated_heatmap(
        z=train_conf_matrix,
        x=["Predicted Non-Diabetes", "Predicted Diabetes"],
        y=["Actual Non-Diabetes", "Actual Diabetes"],
        colorscale="Blues"
    )
    train_conf_matrix_fig['data'][0]['showscale'] = True  # Enable color scale
    train_conf_matrix_fig.update_yaxes(autorange="reversed")
    st.plotly_chart(train_conf_matrix_fig)

    st.write("#### Validation Confusion Matrix")
    val_conf_matrix_fig = ff.create_annotated_heatmap(
        z=val_conf_matrix,
        x=["Predicted Non-Diabetes", "Predicted Diabetes"],
        y=["Actual Non-Diabetes", "Actual Diabetes"],
        colorscale="Blues"
    )
    val_conf_matrix_fig['data'][0]['showscale'] = True  # Enable color scale
    val_conf_matrix_fig.update_yaxes(autorange="reversed")
    st.plotly_chart(val_conf_matrix_fig)     
    
    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_val, model.predict_proba(x_val)[:, 1])
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random chance", line=dict(dash='dash')))
    roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', showlegend=True)
    st.plotly_chart(roc_fig)

if menu == "Predict Diabetes":
    st.title("Predict Diabetes")
    
    with st.form("prediction_form"):
        st.header("Input Patient Data")
        glucose = st.number_input('Glucose', min_value=0, max_value=200, value=117)
        blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=72)
        skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=23)
        insulin = st.number_input('Insulin', min_value=0.0, max_value=846.0, value=30.0)
        bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, value=32.0)
        diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.3725)
        age = st.number_input("Age", min_value=21, max_value=81, value=29)
        submitted = st.form_submit_button("Predict")
        
    if submitted:
        input_data = pd.DataFrame({
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age],
        })
        
        st.subheader('Input Data')
        st.write(input_data)
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.subheader('Prediction Result')
        diabetes_label = np.array(['Non-Diabetes', 'Diabetes'])
        result = diabetes_label[prediction][0]
        st.write(f"Prediction: **{result}**")
        
        if result == "Non-Diabetes":
            st.success("Great news! According to our prediction, you are not at risk of diabetes. Keep up the good work and stay healthy!")
            st.info("You're doing well! Keep up with regular exercise and a balanced diet to maintain your health.")
        else:
            st.warning("It looks like you might be at risk for diabetes. Please consult with a healthcare professional for further advice.")
            st.info("Consider scheduling a visit with your doctor to discuss your health. Early detection and management are key.")
        
        st.subheader('Prediction Probability')
        probability_df = pd.DataFrame(prediction_proba, columns=diabetes_label)
        st.write(probability_df)
        
        fig = px.bar(probability_df, barmode='group')
        st.plotly_chart(fig)
