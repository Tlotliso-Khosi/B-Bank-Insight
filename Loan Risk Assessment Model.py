import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import io
import requests
from urllib.request import urlopen

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Train_data-1-3mTbn73wPHfwMlrH9RoeNlHT3DTNu0.csv"
response = urlopen(url)
data = pd.read_csv(response)

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Data preprocessing
# Convert relevant columns to numeric
numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                'Amount_invested_monthly']

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Extract numeric value from Credit_History_Age (e.g., "6 Years and 4 Months" -> 6.33)
def extract_credit_history_age(age_str):
    try:
        if pd.isna(age_str):
            return np.nan
        parts = age_str.split()
        years = 0
        months = 0
        if 'Years' in age_str or 'Year' in age_str:
            years = float(parts[0])
        if 'Months' in age_str or 'Month' in age_str:
            months_idx = parts.index('and') + 1 if 'and' in parts else 2
            months = float(parts[months_idx])
        return years + (months / 12)
    except:
        return np.nan

if 'Credit_History_Age' in data.columns:
    data['Credit_History_Age_Years'] = data['Credit_History_Age'].apply(extract_credit_history_age)

# Convert Payment_of_Min_Amount to binary
if 'Payment_of_Min_Amount' in data.columns:
    data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})

# Define risk levels based on Credit_Mix
# Assuming Credit_Mix is our target variable for risk assessment
# We'll map: Good -> Low Risk, Standard -> Medium Risk, Bad -> High Risk
if 'Credit_Mix' in data.columns:
    risk_mapping = {'Good': 0, 'Standard': 1, 'Bad': 2}
    data['Risk_Level'] = data['Credit_Mix'].map(risk_mapping)
    print("\nRisk level distribution:")
    print(data['Risk_Level'].value_counts())

# Define features and target
# Exclude non-predictive columns
exclude_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan', 
                'Credit_Mix', 'Risk_Level', 'Payment_Behaviour', 'Credit_History_Age']
categorical_cols = ['Occupation']

# Get all numeric columns
numeric_features = [col for col in data.columns if col not in exclude_cols + categorical_cols 
                   and col != 'Risk_Level' and data[col].dtype in ['int64', 'float64']]

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Handle missing values
data = data.dropna(subset=['Risk_Level'])  # Drop rows with missing target
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

# Split the data
X = data[numeric_features + categorical_cols]
y = data['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nTraining the model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
feature_names = numeric_features + [f"{col}_{cat}" for col, cats in 
                                   zip([col for _, _, col in preprocessor.transformers_], 
                                       model.named_steps['preprocessor'].transformers_[1][1].categories_) 
                                   for cat in cats[0]]

importances = model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Ranking:")
for i, idx in enumerate(indices[:10]):  # Print top 10 features
    if i < len(feature_names):
        print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

# Define risk levels and their meanings
risk_levels = {
    0: {
        'name': 'Low Risk',
        'description': 'Applicant has good credit history and financial stability. Loan approval recommended.',
        'approval_chance': '85-100%'
    },
    1: {
        'name': 'Medium Risk',
        'description': 'Applicant has average credit history with some concerns. Careful consideration needed.',
        'approval_chance': '50-85%'
    },
    2: {
        'name': 'High Risk',
        'description': 'Applicant has poor credit history or financial instability. Loan approval not recommended.',
        'approval_chance': '0-50%'
    }
}

# Function to predict risk and provide detailed analysis
def predict_loan_risk(data_point, model=model, feature_names=feature_names, risk_levels=risk_levels):
    """
    Predict loan risk and provide detailed analysis
    
    Parameters:
    data_point (dict): Dictionary with applicant information
    
    Returns:
    dict: Risk assessment results
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([data_point])
    
    # Make prediction
    risk_level = model.predict(input_df)[0]
    risk_proba = model.predict_proba(input_df)[0]
    
    # Calculate risk percentage (probability of being high risk)
    risk_percentage = risk_proba[2] * 100 if len(risk_proba) > 2 else 0
    
    # Get feature contributions
    # For this, we'll use the feature importances multiplied by the feature values
    feature_contributions = {}
    
    # Get preprocessed features
    preprocessed = model.named_steps['preprocessor'].transform(input_df)
    
    # Calculate contribution for each feature
    for i, feature in enumerate(feature_names):
        if i < preprocessed.shape[1]:
            importance = model.named_steps['classifier'].feature_importances_[i]
            # Normalize the preprocessed value for better interpretation
            value = preprocessed[0, i]
            contribution = importance * abs(value) * 100  # Convert to percentage
            feature_contributions[feature] = contribution
    
    # Sort contributions by importance
    sorted_contributions = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    top_factors = sorted_contributions[:5]  # Top 5 contributing factors
    
    return {
        'risk_level': risk_levels[risk_level]['name'],
        'risk_percentage': f"{risk_percentage:.2f}%",
        'risk_description': risk_levels[risk_level]['description'],
        'approval_chance': risk_levels[risk_level]['approval_chance'],
        'top_contributing_factors': top_factors
    }

# Example usage
example_applicant = {
    'Age': 35,
    'Annual_Income': 60000,
    'Monthly_Inhand_Salary': 4500,
    'Num_Bank_Accounts': 2,
    'Num_Credit_Card': 3,
    'Interest_Rate': 12,
    'Num_of_Loan': 1,
    'Delay_from_due_date': 5,
    'Num_of_Delayed_Payment': 2,
    'Changed_Credit_Limit': 1.5,
    'Num_Credit_Inquiries': 3,
    'Outstanding_Debt': 15000,
    'Credit_Utilization_Ratio': 25,
    'Total_EMI_per_month': 1200,
    'Amount_invested_monthly': 500,
    'Occupation': 'Engineer'
}

print("\nExample Risk Assessment:")
risk_assessment = predict_loan_risk(example_applicant)
print(f"Risk Level: {risk_assessment['risk_level']}")
print(f"Risk Percentage: {risk_assessment['risk_percentage']}")
print(f"Description: {risk_assessment['risk_description']}")
print(f"Approval Chance: {risk_assessment['approval_chance']}")
print("\nTop Contributing Factors:")
for factor, contribution in risk_assessment['top_contributing_factors']:
    print(f"- {factor}: {contribution:.2f}%")

# Save the model
model_filename = 'loan_risk_assessment_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

# Function to load the model
def load_model(filename=model_filename):
    """Load the trained model"""
    return joblib.load(filename)

print("\nTo use the saved model:")
print("loaded_model = load_model('loan_risk_assessment_model.pkl')")
print("risk_assessment = predict_loan_risk(applicant_data, loaded_model)")