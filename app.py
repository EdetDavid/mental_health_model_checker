import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Set page configuration
st.set_page_config(
    page_title="Student Mental Health Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_PATH = "models/"
DATA_PATH = "data/student_mental_health.csv"

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Function to load models
def load_models(model_path=MODEL_PATH):
    """Load trained models from disk."""
    try:
        with open(os.path.join(model_path, "best_regression_model.pkl"), "rb") as f:
            reg_model = pickle.load(f)
        with open(os.path.join(model_path, "best_classification_model.pkl"), "rb") as f:
            cls_model = pickle.load(f)
        with open(os.path.join(model_path, "preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)
        return reg_model, cls_model, preprocessor
    except FileNotFoundError:
        st.error("Models not found. Please train the models first.")
        return None, None, None

# Function to train and save models
def train_and_save_models(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Train models and save them to disk."""
    # Load data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        return False

    # Define features and target
    X = df.drop(columns=['mental_health_index', 'mental_health_category'])
    y_reg = df['mental_health_index']
    y_cls = df['mental_health_category']

    # Split data
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create and train Random Forest models (best performers)
    rf_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    rf_cls = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train models
    with st.spinner("Training regression model..."):
        rf_reg.fit(X_train, y_train_reg)
        # Evaluate
        y_pred_reg = rf_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        st.write(f"Regression Model RMSE: {rmse:.2f}")

    with st.spinner("Training classification model..."):
        rf_cls.fit(X_train, y_train_cls)
        # Evaluate
        y_pred_cls = rf_cls.predict(X_test)
        accuracy = accuracy_score(y_test_cls, y_pred_cls)
        st.write(f"Classification Model Accuracy: {accuracy:.2f}")
        st.text(classification_report(y_test_cls, y_pred_cls))

    # Save models
    with open(os.path.join(model_path, "best_regression_model.pkl"), "wb") as f:
        pickle.dump(rf_reg, f)
    with open(os.path.join(model_path, "best_classification_model.pkl"), "wb") as f:
        pickle.dump(rf_cls, f)
    with open(os.path.join(model_path, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    
    st.success("Models trained and saved successfully!")
    return True

# Function to make predictions
def predict_mental_health(student_data, reg_model, cls_model):
    """Predict mental health outcomes for a student."""
    # Convert input to DataFrame
    student_df = pd.DataFrame([student_data])
    
    results = {}
    
    # Get regression prediction
    mh_index = reg_model.predict(student_df)[0]
    results["mental_health_index"] = mh_index
    
    # Get classification prediction
    mh_category = cls_model.predict(student_df)[0]
    category_probs = cls_model.predict_proba(student_df)[0]
    classes = cls_model.classes_
    results["mental_health_category"] = mh_category
    results["category_probabilities"] = {classes[i]: category_probs[i] for i in range(len(classes))}
    
    return results

# Main function
def main():
    # App title
    st.title("🧠 Student Mental Health Prediction System")
    st.write("This system helps university counselors assess potential mental health concerns for students.")
    
    # Sidebar for model training
    st.sidebar.header("Model Management")
    if st.sidebar.button("Train Models"):
        train_and_save_models()
    
    # Load trained models
    reg_model, cls_model, preprocessor = load_models()
    
    if reg_model is None or cls_model is None:
        st.warning("Please train the models first by clicking the 'Train Models' button in the sidebar.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Make Prediction", "Explore Data", "About"])
    
    with tab1:
        # Create form for student information
        with st.form("student_info_form"):
            st.subheader("Student Information")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            # Demographics
            with col1:
                st.subheader("Demographics")
                age = st.slider("Age", 17, 30, 20)
                gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary"])
                international = st.selectbox("International Student", options=["Yes", "No"])
                year = st.selectbox("Year of Study", options=[1, 2, 3, 4, 5])
                marital_status = st.selectbox("Marital Status", 
                                            options=["Single", "In a relationship", "Married", "Divorced"])
                
                # Academic factors
                st.subheader("Academic Factors")
                gpa = st.slider("GPA", 0.0, 4.0, 3.0, 0.1)
                major = st.selectbox("Major", options=["Engineering", "Arts", "Science", "Business", "Medicine", 
                                                     "Education", "Social Sciences", "Computer Science"])
                course_load = st.slider("Course Load (# of courses)", 1, 8, 5)
            
            # Health and other factors
            with col2:
                # Health indicators
                st.subheader("Health Indicators")
                sleep = st.slider("Sleep Hours (daily average)", 3.0, 10.0, 7.0, 0.5)
                exercise = st.selectbox("Exercise (weekly)", 
                                     options=["None", "1-2 days", "3-4 days", "5+ days"])
                
                # Mental health screening
                st.subheader("Mental Health Screening")
                depression = st.slider("Depression Score (PHQ-9)", 0, 27, 5, 
                                    help="0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-19: Moderately Severe, 20-27: Severe")
                anxiety = st.slider("Anxiety Score (GAD-7)", 0, 21, 5, 
                                 help="0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-21: Severe")
                stress = st.selectbox("Stress Level", options=["Low", "Moderate", "High", "Severe"])
                
                # Support systems
                st.subheader("Support Systems")
                counseling = st.selectbox("Previous Counseling", options=["Yes", "No"])
                family_support = st.selectbox("Family Support", options=["Poor", "Fair", "Good", "Excellent"])
                friend_support = st.selectbox("Friend Support", options=["Poor", "Fair", "Good", "Excellent"])
            
            # Additional factors in a single column
            st.subheader("Additional Factors")
            col1, col2 = st.columns(2)
            
            with col1:
                financial_stress = st.selectbox("Financial Stress Level", 
                                            options=["None", "Low", "Moderate", "High", "Severe"])
                living_condition = st.selectbox("Living Condition", 
                                            options=["Poor", "Fair", "Good", "Excellent"])
            
            with col2:
                screen_time = st.selectbox("Daily Screen Time", 
                                        options=["1-2 hours", "3-5 hours", "6-8 hours", "9+ hours"])
                social_media = st.selectbox("Daily Social Media Usage", 
                                         options=["Less than 1 hour", "1-2 hours", "3-4 hours", "5+ hours"])
            
            # Submit button
            submitted = st.form_submit_button("Generate Prediction")
        
        # When form is submitted
        if submitted:
            # Create student data dictionary
            student_data = {
                "age": age,
                "gender": gender,
                "international": international,
                "year_of_study": year,
                "marital_status": marital_status,
                "gpa": gpa,
                "major": major,
                "course_load": course_load,
                "sleep_hours": sleep,
                "exercise_weekly": exercise,
                "depression_score": depression,
                "anxiety_score": anxiety,
                "stress_level": stress,
                "counseling_before": counseling,
                "family_support": family_support,
                "friend_support": friend_support,
                "financial_stress": financial_stress,
                "living_condition": living_condition,
                "screen_time_daily": screen_time,
                "social_media_daily": social_media
            }
            
            # Get prediction
            with st.spinner("Generating prediction..."):
                prediction = predict_mental_health(student_data, reg_model, cls_model)
            
            # Display results
            st.header("Mental Health Assessment Results")
            
            # Display mental health index
            mh_index = prediction["mental_health_index"]
            st.subheader(f"Mental Health Index: {mh_index:.1f}/100")
            
            # Create a gauge chart for the mental health index
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh([0], [100], color="lightgray", height=0.5)
            
            # Choose color based on index value (red->yellow->green)
            if mh_index < 40:
                color = 'red'
            elif mh_index < 60:
                color = 'orange'
            else:
                color = 'green'
                
            ax.barh([0], [mh_index], color=color, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0\nPoor", "25", "50\nAverage", "75", "100\nExcellent"])
            st.pyplot(fig)
            
            # Display mental health category
            st.subheader(f"Mental Health Category: {prediction['mental_health_category']}")
            
            # Display category probabilities
            st.write("Probability Breakdown:")
            probs = prediction["category_probabilities"]
            
            # Create a bar chart for probabilities
            fig, ax = plt.subplots(figsize=(8, 3))
            categories = list(probs.keys())
            probabilities = list(probs.values())
            colors = ['red', 'orange', 'green']
            ax.bar(categories, probabilities, color=colors[:len(categories)])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Mental Health Category Probabilities')
            
            # Add percentage labels on bars
            for i, v in enumerate(probabilities):
                ax.text(i, v + 0.05, f"{v:.1%}", ha='center')
                
            st.pyplot(fig)
                
            # Recommendation section
            st.subheader("Recommendations")
            if prediction["mental_health_category"] == "Poor":
                st.error("⚠️ This student shows signs of significant mental health concerns. Consider immediate follow-up and referral to psychological services.")
            elif prediction["mental_health_category"] == "Average":
                st.warning("This student shows moderate risk. Regular check-ins and providing resources for support would be beneficial.")
            else:
                st.success("This student appears to be maintaining good mental health. Continue to provide preventive resources.")
                
            # Risk factors analysis
            st.subheader("Key Risk Factors")
            risk_factors = []
            if sleep < 6:
                risk_factors.append("Insufficient sleep (< 6 hours)")
            if exercise == "None":
                risk_factors.append("Limited physical activity")
            if financial_stress in ["High", "Severe"]:
                risk_factors.append("High financial stress")
            if family_support in ["Poor", "Fair"]:
                risk_factors.append("Limited family support")
            if friend_support in ["Poor", "Fair"]:
                risk_factors.append("Limited friend support")
            if depression > 9:
                risk_factors.append("Elevated depression score")
            if anxiety > 9:
                risk_factors.append("Elevated anxiety score")
            if stress in ["High", "Severe"]:
                risk_factors.append("High stress levels")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No major risk factors identified.")
    
    with tab2:
        st.header("Data Exploration")
        try:
            # Load and display data
            df = pd.read_csv(DATA_PATH)
            st.write(f"Dataset contains {len(df)} student records.")
            
            # Display basic statistics
            if st.checkbox("Show Data Statistics"):
                st.subheader("Basic Statistics")
                st.write(df.describe())
            
            # Display correlation heatmap
            if st.checkbox("Show Correlation Heatmap"):
                st.subheader("Correlation Heatmap")
                numerical_df = df.select_dtypes(include=['int64', 'float64'])
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)
            
            # Distribution of mental health categories
            st.subheader("Mental Health Categories Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x='mental_health_category', data=df, ax=ax)
            plt.title('Distribution of Mental Health Categories')
            plt.ylabel('Count')
            st.pyplot(fig)
            
            # Feature importance
            if st.checkbox("Show Feature Importance"):
                st.subheader("Feature Importance")
                # Load models to get feature importance
                reg_model, _, _ = load_models()
                if reg_model:
                    # Get feature names after preprocessing
                    all_features = list(df.drop(columns=['mental_health_index', 'mental_health_category']).columns)
                    
                    # We're using a Random Forest model which has feature_importances_
                    try:
                        importances = reg_model.named_steps['model'].feature_importances_
                        # Use the first 10 importances if the full list is too long
                        importances_subset = importances[:10] if len(importances) > 10 else importances
                        features_subset = all_features[:10] if len(all_features) > 10 else all_features
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=importances_subset, y=features_subset, ax=ax)
                        plt.title('Feature Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except:
                        st.write("Could not extract feature importance from the model.")
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    with tab3:
        st.header("About this System")
        st.write("""
        ## Student Mental Health Prediction System
        
        This system uses machine learning to predict student mental health outcomes based on various factors
        including demographics, academic performance, lifestyle, and support systems.
        
        ### How it works
        
        1. **Data Collection**: Information about a student is gathered through the form
        2. **Prediction**: The system uses trained machine learning models to predict:
           - Mental Health Index: A continuous score from 0-100
           - Mental Health Category: Poor, Average, or Good
        3. **Analysis**: The system identifies potential risk factors and provides recommendations
        
        ### Models Used
        
        The system uses Random Forest algorithms for both regression (predicting the mental health index) 
        and classification (predicting the mental health category).
        
        ### Disclaimer
        
        This system is intended as a screening aid for university counseling services. It should not 
        be used as a diagnostic system or as a replacement for professional assessment.
        
        ### Privacy Notice
        
        Ensure all data is handled in compliance with your institution's privacy policies and relevant 
        regulations (FERPA, HIPAA, etc.).
        """)

# Run the app
if __name__ == "__main__":
    main()
