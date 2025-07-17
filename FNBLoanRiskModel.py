import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the FNB dataset
try:
    data = pd.read_csv('FNB_Train_data.csv')
    print("FNB Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
except Exception as e:
    print(f"Error loading FNB dataset: {e}")
    exit()

# Data preprocessing
numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                'Amount_invested_monthly', 'Monthly_Balance']

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Extract numeric value from Credit_History_Age
def extract_credit_history_age(age_str):
    try:
        if pd.isna(age_str) or age_str == 'NA':
            return np.nan
        parts = str(age_str).split()
        years = 0
        months = 0
        if 'Years' in str(age_str) or 'Year' in str(age_str):
            years = float(parts[0])
        if 'Months' in str(age_str) or 'Month' in str(age_str):
            months_idx = parts.index('and') + 1 if 'and' in parts else 2
            if months_idx < len(parts):
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
if 'Credit_Mix' in data.columns:
    risk_mapping = {'Good': 0, 'Standard': 1, 'Bad': 2}
    data['Risk_Level'] = data['Credit_Mix'].map(risk_mapping)
    # Handle missing values
    data['Risk_Level'] = data['Risk_Level'].fillna(1)
    print("\nRisk level distribution:")
    print(data['Risk_Level'].value_counts())

# Define features
numeric_features = [col for col in numeric_cols + ['Credit_History_Age_Years', 'Payment_of_Min_Amount'] 
                   if col in data.columns]
categorical_cols = ['Occupation'] if 'Occupation' in data.columns else []

# Handle missing values
data = data.dropna(subset=['Risk_Level'])
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
if categorical_cols:
    data[categorical_cols] = data[categorical_cols].fillna('Unknown')

# Split the data
X = data[numeric_features + categorical_cols]
y = data['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ] if categorical_cols else [
        ('num', StandardScaler(), numeric_features)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nTraining the FNB model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFNB Model Accuracy: {accuracy:.4f}")

# Save the model
model_filename = 'FNB_loan_risk_assessment_model.pkl'
joblib.dump(model, model_filename)
print(f"\nFNB Model saved as {model_filename}")
