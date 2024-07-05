from urllib import response
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import KFold, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xml.etree.ElementTree as ET
import plotly.figure_factory as ff
from streamlit_option_menu import option_menu
from knowledge_base import KnowledgeBase
from fuzzy_system import create_fuzzy_system
# Load model and data
model = pickle.load(open('random_forest_model.pkl', 'rb'))

x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
x_val = pd.read_csv('X_val.csv')
y_val = pd.read_csv('y_val.csv').squeeze()

@st.cache_resource
def init_knowledge_base():
    kb = KnowledgeBase("mongodb+srv://rabirabi:Rabirabi80@cluster0.ylk5353.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

    if kb.get_all_topics() == []:
        kb.initialize_json('diabetes_knowledge_base.json')
    
    return kb

kb = init_knowledge_base()

fuzzy_system = create_fuzzy_system()

def predict_diabetes_with_fuzzy(input_data,model,fuzzy_system):

    model_prediction = model.predict_proba(input_data)[0][1]

    # Fuzzy logic prediction
    fuzzy_system.input['glucose'] = input_data['Glucose'].values[0]
    fuzzy_system.input['bmi'] = input_data['BMI'].values[0]
    fuzzy_system.input['age'] = input_data['Age'].values[0]

    fuzzy_system.compute()

    fuzzy_risk = fuzzy_system.output['risk']

    combine_risk = (0.7 * model_prediction * 100) + (0.3 * fuzzy_risk)

    return combine_risk, model_prediction * 100 , fuzzy_risk


# Sidebar for navigation
with st.sidebar:
    menu = option_menu(
        menu_title= "Navigation",
        options=['Home', "About", "Diabetes Information", "Model Performance", "Data Exploration", "Predict Diabetes"],
        default_index=0
    )

# Homepage

if menu == "Home":
    st.title("Diabetes Prediction and Information Dashboard")
    st.image("sample.jpg", caption='Diabetes Awareness', use_column_width=True)
    st.write("""
    Welcome to the Diabetes Prediction and Information Dashboard.
    This tool combines advanced machine learning with fuzzy logic and fuzzy string matching to:
    
    1. Predict the likelihood of diabetes based on input parameters using a Random Forest model.
    2. Provide answers to common questions about diabetes using a fuzzy logic-based system.
    3. Intelligently match user queries to our knowledge base using fuzzy string matching.
    
    Use the sidebar to navigate through the different sections of the dashboard:
    
    - **About**: Learn more about the project and the technologies used.
    - **Diabetes Information**: Ask questions about diabetes and get informative answers.
    - **Model Performance**: Explore the performance metrics of our prediction model.
    - **Data Exploration**: Visualize and analyze the dataset used for prediction.
    - **Predict Diabetes**: Input patient data and get a prediction on diabetes likelihood.
    
    Whether you're a healthcare professional, researcher, or simply interested in learning more about diabetes, 
    this dashboard offers valuable insights and tools to assist you.
    """)

# About Page
if menu == "About":
    with st.container():
        st.title("About This Project")
        st.image("sample.jpg", caption='Diabetes Research', width=360)
        st.write("""
        This project is part of the Decision Support System Course, combining machine learning prediction 
        with an intelligent question-answering system for diabetes information.

        ### :evergreen_tree: Random Forest for Prediction
        We use a Random Forest algorithm to predict the likelihood of diabetes. Random Forest is an ensemble 
        learning method that constructs multiple decision trees and outputs the mode of the classes for 
        classification tasks.

        Key steps in the Random Forest algorithm:
        1. **Bootstrapping**: Creating random subsets of the original data with replacement.
        2. **Building Trees**: Constructing a decision tree for each subset.
        3. **Feature Selection**: Considering a random subset of features at each split.
        4. **Aggregation**: Making the final prediction by aggregating predictions from all trees.

        ### :brain: Fuzzy Logic for Risk Assessment
        We implement a fuzzy logic-based system to assess diabetes risk based on key factors like glucose levels, 
        BMI, and age. This system:
        
        1. Defines fuzzy sets for input variables.
        2. Establishes fuzzy rules to determine risk levels.
        3. Combines with the Random Forest model for a more nuanced risk assessment.

        ### :mag: Fuzzy String Matching for Information Retrieval
        Our system uses fuzzy string matching to intelligently answer questions about diabetes:
        
        1. Matches user queries to our knowledge base using algorithms like Levenshtein distance.
        2. Handles variations in spelling, phrasing, and word order.
        3. Returns the most relevant information even when queries don't exactly match predefined questions.

        ### Benefits of Our Approach
        - **Comprehensive Analysis**: Combines predictive modeling with informational support.
        - **Flexible Query Handling**: Understands and responds to a variety of question phrasings.
        - **User-Friendly Interface**: Easy-to-use dashboard for both prediction and information retrieval.
        - **Robust Information Retrieval**: Finds relevant information even with imperfect user queries.
        - **Continual Improvement**: The knowledge base can be expanded over time for more comprehensive coverage.

        Use the navigation menu to explore model performance, make predictions, or ask questions about diabetes.
        """)

# Diabetes Information Page
if menu == "Diabetes Information":
    st.title("Diabetes Information Assistant")
    st.write("""
    Welcome to the Diabetes Information section. Here you can ask questions about diabetes and get information 
    from our curated knowledge base.
    """)
    submenu = option_menu(
        menu_title="",
        options=['Ask the Question', 'Add New Information'],
        default_index=0,
        key='submenu'
    )
    if submenu == 'Ask the Question':
        user_input = st.text_input("Ask a Question About Diabetes:")

        if user_input:
            response = kb.get_diabetes_info(user_input)
            st.write(f"{response} ")
    
        st.write("""Our system uses fuzzy logic to match your questions with the most relevant information available. 
        This allows for natural language queries and provides answers even when questions are not asked in a standardized format.""")
        st.info("""
    Please note: This information is for educational purposes only and should not be considered medical advice. 
    Always consult with a healthcare professional for medical concerns. \n
    Remember: While this system can provide general information, it's not a substitute for professional medical advice. 
    If you have concerns about diabetes, please consult with a healthcare provider.
    """)

    # Add some example questions to guide users
        st.subheader("Example questions you can ask:")
        st.write("- What are the symptoms of diabetes?")
        st.write("- How can I prevent diabetes?")
        st.write("- What are the different types of diabetes?")
        st.write("- How does exercise affect diabetes?")
    elif submenu == 'Add New Information':
        st.write("Add New Information Content")
        new_topic =st.text_input("New Topic:")
        new_info = st.text_area("Information:")
        if st.button("Add to Knowledge Base"):
            if new_topic and new_info :
                kb.add_entry(new_topic, new_info)
                st.success("Information Added successfully")
            else:
                st.error("Please Provide bith a topic and information")
  

# Model Performance Page
if menu == "Model Performance":
    st.title("Model Performance Metrics")
    st.write("""
    This section provides a detailed analysis of our Random Forest model's performance in predicting diabetes. 
    Explore various metrics, visualizations, and insights into how well our model performs on both training and 
    validation data.
    """)

    # Predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Metrics calculation
    def calculate_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_pred),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)

    with st.expander("Key Performance Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Metrics")
            for metric, value in train_metrics.items():
                if metric != "Confusion Matrix":
                    st.write(f"**{metric}**: {value:.4f}")
        with col2:
            st.subheader("Validation Metrics")
            for metric, value in val_metrics.items():
                if metric != "Confusion Matrix":
                    st.write(f"**{metric}**: {value:.4f}")

    with st.expander("Confusion Matrices"):
        def plot_confusion_matrix(conf_matrix, title):
            fig = ff.create_annotated_heatmap(
                z=conf_matrix,
                x=["Predicted Non-Diabetes", "Predicted Diabetes"],
                y=["Actual Non-Diabetes", "Actual Diabetes"],
                colorscale="Blues"
            )
            fig.update_layout(title=title)
            fig.update_yaxes(autorange="reversed")
            return fig

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_confusion_matrix(train_metrics["Confusion Matrix"], "Training Confusion Matrix"))
        with col2:
            st.plotly_chart(plot_confusion_matrix(val_metrics["Confusion Matrix"], "Validation Confusion Matrix"))

    with st.expander("ROC Curve"):
        fpr, tpr, _ = roc_curve(y_val, y_val_pred)
        fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        st.plotly_chart(fig)

    with st.expander("Feature Correlation"):
        st.subheader("Feature Correlation Heatmap")
        corr = x_train.corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(width=700, height=700)
        st.plotly_chart(fig)

    with st.expander("Feature Importance"):
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_imp = pd.DataFrame(sorted(zip(importances, x_train.columns)), columns=['Value','Feature'])
        fig = px.bar(feature_imp, x='Value', y='Feature', orientation='h', title='Feature Importances')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)

    with st.expander("Error Analysis"):
        st.subheader("Error Analysis")

        # Get misclassified instances
        misclassified = x_val[y_val != y_val_pred].copy()
        misclassified['True_Label'] = y_val[y_val != y_val_pred]
        misclassified['Predicted_Label'] = y_val_pred[y_val != y_val_pred]

        st.write("Sample of Misclassified Instances:")
        st.write(misclassified.head())

        # Analyze feature distributions for misclassified instances
        feature = st.selectbox("Select feature for misclassification analysis", x_val.columns)
        fig = px.histogram(misclassified, x=feature, color='True_Label', marginal="box", 
                       title=f"Distribution of {feature} for Misclassified Instances")
        st.plotly_chart(fig)

    with st.expander("Cross-Validation Results"):
        st.subheader("Cross-Validation Results")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, x_train, y_train, cv=kf)
        cv_df = pd.DataFrame({'Fold': range(1, len(cv_scores)+1), 'Score': cv_scores})
        fig = px.box(cv_df, y='Score', points="all", title='Cross-Validation Scores')
        st.plotly_chart(fig)

        st.write(f"Mean CV Score: {cv_scores.mean():.4f}")
        st.write(f"Standard Deviation: {cv_scores.std():.4f}")

if menu == 'Data Exploration':
    st.title('Exploratory Data Analysis')
    st.write("""
    Dive into the dataset used for training our diabetes prediction model. This section allows you to visualize 
    the distribution of various features and their relationship with diabetes outcomes.
    """)

    feature = st.selectbox("Select Feature for Histogram", x_train.columns)

    st.subheader(f"Histogram of {feature}")
    fig = px.histogram(x_train, x=feature, color=y_train.values, marginal="box")
    st.plotly_chart(fig)

    # Box plot
    st.subheader(f"Box Plot of {feature} by Diabetes Status")
    fig = px.box(x_train, y=feature, color=y_train.values, points="all")
    st.plotly_chart(fig)

    # Target variable distribution
    st.subheader("Distribution of Target Variable")
    fig = px.pie(values=y_train.value_counts().values, names=y_train.value_counts().index, title='Distribution status Diabetes')
    st.plotly_chart(fig)

if menu == "Predict Diabetes":
    st.title("Predict Diabetes")
    st.write("""
    This tool combines a Random Forest model with a fuzzy logic system to predict the likelihood of diabetes. 
    The fuzzy system focuses on key factors: Glucose level, BMI, and Age.
    """)

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
            'Age': [age]
        })

        st.subheader('Input Data')
        st.write(input_data)

        combined_risk, model_risk, fuzzy_risk = predict_diabetes_with_fuzzy(input_data, model, fuzzy_system)

        st.subheader('Prediction Result')
        st.write(f"Combined Risk: {combined_risk:.2f}%")
        st.write(f"Random Forest Model Risk: {model_risk:.2f}%")
        st.write(f"Fuzzy Logic Risk: {fuzzy_risk:.2f}%")

        if combined_risk < 50:
            st.success("Based on our combined analysis, you appear to have a lower risk of diabetes. However, always consult with a healthcare professional for a proper evaluation.")
        else:
            st.warning("Based on our combined analysis, you may have an elevated risk of diabetes. Please consult with a healthcare professional for a thorough evaluation and advice.")
        
        # Visualization
        fig = go.Figure(go.Bar(
            x=['Combined Risk', 'Random Forest Risk', 'Fuzzy Logic Risk'],
            y=[combined_risk, model_risk, fuzzy_risk],
            text=[f'{combined_risk:.1f}%', f'{model_risk:.1f}%', f'{fuzzy_risk:.1f}%'],
            textposition='auto',
        ))
        fig.update_layout(title='Risk Comparison', yaxis_title='Risk (%)')
        st.plotly_chart(fig)

        st.info("Note: This prediction combines machine learning (Random Forest) with fuzzy logic for a more nuanced assessment. The fuzzy logic system takes into account the imprecision in medical data and diagnostic boundaries.")