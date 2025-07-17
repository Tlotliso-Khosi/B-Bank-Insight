import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the B-Bank dataset
try:
    data = pd.read_csv('B-Bank_Train_data.csv')
    print("B-Bank Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
except Exception as e:
    print(f"Error loading B-Bank dataset: {e}")
    # If B-Bank dataset doesn't exist, create it from other datasets
    try:
        print("Creating B-Bank dataset from other sources...")
        datasets = []
        
        # Load SB dataset
        try:
            sb_data = pd.read_csv('SB_Train_data.csv')
            sb_data['Source_Bank'] = 'SB'
            datasets.append(sb_data)
            print(f"✓ SB Dataset loaded: {sb_data.shape}")
        except Exception as e:
            print(f"✗ Error loading SB dataset: {e}")

        # Load PB dataset
        try:
            pb_data = pd.read_csv('PB_Train_data.csv')
            pb_data['Source_Bank'] = 'PB'
            datasets.append(pb_data)
            print(f"✓ PB Dataset loaded: {pb_data.shape}")
        except Exception as e:
            print(f"✗ Error loading PB dataset: {e}")

        # Load FNB dataset
        try:
            fnb_data = pd.read_csv('FNB_Train_data.csv')
            fnb_data['Source_Bank'] = 'FNB'
            datasets.append(fnb_data)
            print(f"✓ FNB Dataset loaded: {fnb_data.shape}")
        except Exception as e:
            print(f"✗ Error loading FNB dataset: {e}")

        # Combine all datasets
        if datasets:
            data = pd.concat(datasets, ignore_index=True)
            print(f"✓ B-Bank Combined Dataset created: {data.shape}")
        else:
            print("✗ No datasets available for B-Bank model")
            exit()
    except Exception as e2:
        print(f"Error creating B-Bank dataset: {e2}")
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
    print("\nB-Bank Risk level distribution:")
    print(data['Risk_Level'].value_counts())

# Define features (including Source_Bank for B-Bank model if it exists)
numeric_features = [col for col in numeric_cols + ['Credit_History_Age_Years', 'Payment_of_Min_Amount'] 
                   if col in data.columns]
categorical_cols = ['Occupation']
if 'Source_Bank' in data.columns:
    categorical_cols.append('Source_Bank')

# Handle missing values
data = data.dropna(subset=['Risk_Level'])
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna('Unknown')

print(f"Final dataset shape for B-Bank model: {data.shape}")
print(f"Features: {len(numeric_features)} numeric, {len(categorical_cols)} categorical")

# Split the data
X = data[numeric_features + categorical_cols]
y = data['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create enhanced preprocessing pipeline for B-Bank
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_  numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create enhanced B-Bank model with more sophisticated parameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,  # More trees for better performance
        max_depth=15,      # Deeper trees for complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1  # Use all available cores
    ))
])

print("\nTraining the B-Bank master model...")
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nB-Bank Master Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
feature_names = numeric_features + categorical_cols
importances = model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 Most Important Features for B-Bank Model:")
for i, idx in enumerate(indices[:10]):
    if i < len(feature_names):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Save the B-Bank master model
model_filename = 'B-Bank_loan_risk_assessment_model.pkl'
joblib.dump(model, model_filename)
print(f"\nB-Bank Master Model saved as {model_filename}")

# Display model statistics
print(f"\nB-Bank Master Model Statistics:")
print(f"- Total training samples: {len(X_train)}")
print(f"- Total features: {len(numeric_features + categorical_cols)}")
if 'Source_Bank' in data.columns:
    print(f"- Data sources: {data['Source_Bank'].value_counts().to_dict()}")
print(f"- Model accuracy: {accuracy:.4f}")
print("\nB-Bank master model training completed successfully!")
