import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_credit_history_age(age_str):
    """Extract numeric value from Credit_History_Age"""
    try:
        if pd.isna(age_str) or age_str == 'NA' or str(age_str).strip() == '':
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

def train_bank_model(csv_file, model_name, output_file):
    """Train a model for a specific bank"""
    print(f"\n🔄 Training {model_name} model...")
    
    try:
        # Load data
        data = pd.read_csv(csv_file)
        print(f"✓ Dataset loaded: {data.shape}")
        
        # Data preprocessing
        numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                        'Amount_invested_monthly', 'Monthly_Balance']
        
        # Convert to numeric
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Process credit history
        if 'Credit_History_Age' in data.columns:
            data['Credit_History_Age_Years'] = data['Credit_History_Age'].apply(extract_credit_history_age)
        
        # Binary conversion
        if 'Payment_of_Min_Amount' in data.columns:
            data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})
        
        # Risk level mapping
        if 'Credit_Mix' in data.columns:
            risk_mapping = {'Good': 0, 'Standard': 1, 'Bad': 2}
            data['Risk_Level'] = data['Credit_Mix'].map(risk_mapping)
            data['Risk_Level'] = data['Risk_Level'].fillna(1)
        else:
            print(f"✗ No Credit_Mix column found in {model_name} dataset")
            return False
        
        # Feature selection
        numeric_features = [col for col in numeric_cols + ['Credit_History_Age_Years', 'Payment_of_Min_Amount'] 
                           if col in data.columns]
        categorical_cols = ['Occupation'] if 'Occupation' in data.columns else []
        
        # Handle missing values
        data = data.dropna(subset=['Risk_Level'])
        if len(data) < 50:
            print(f"✗ Insufficient data for {model_name} model")
            return False
        
        # Fill missing values
        for col in numeric_features:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        if categorical_cols:
            for col in categorical_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown')
        
        # Prepare features
        available_features = [col for col in numeric_features + categorical_cols if col in data.columns]
        X = data[available_features]
        y = data['Risk_Level']
        
        # Check target variety
        if len(y.unique()) < 2:
            print(f"✗ Insufficient target variety for {model_name} model")
            return False
        
        # Preprocessing pipeline
        numeric_available = [col for col in numeric_features if col in data.columns]
        categorical_available = [col for col in categorical_cols if col in data.columns]
        
        transformers = []
        if numeric_available:
            transformers.append(('num', RobustScaler(), numeric_available))
        if categorical_available:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_available))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Create model
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train base model
        base_model.fit(X_train, y_train)
        
        # Calibrate for better probabilities
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=min(3, len(np.unique(y_train))))
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = calibrated_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ {model_name} Model trained - Accuracy: {accuracy:.4f}")
        
        # Save model and metadata
        model_path = os.path.join(MODEL_DIR, output_file)
        joblib.dump(calibrated_model, model_path)
        
        # Save metadata
        metadata = {
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_names': available_features,
            'trained_at': datetime.now().isoformat(),
            'model_name': model_name
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"✓ {model_name} model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error training {model_name} model: {e}")
        return False

def create_bbank_combined_model():
    """Create B-Bank combined model from all bank datasets"""
    print(f"\n🔄 Creating B-Bank combined model...")
    
    try:
        # Load all datasets
        datasets = []
        bank_files = {
            'SB': 'SB_Train_data.csv',
            'PB': 'PB_Train_data.csv',
            'FNB': 'FNB_Train_data.csv'
        }
        
        for bank, filename in bank_files.items():
            try:
                data = pd.read_csv(filename)
                data['Source_Bank'] = bank
                datasets.append(data)
                print(f"✓ {bank} data loaded: {data.shape}")
            except Exception as e:
                print(f"⚠️ Could not load {bank} data: {e}")
        
        if not datasets:
            print("✗ No datasets available for B-Bank model")
            return False
        
        # Combine datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        if 'Customer_ID' in combined_data.columns:
            combined_data = combined_data.drop_duplicates(subset=['Customer_ID'], keep='first')
        
        print(f"✓ Combined dataset created: {combined_data.shape}")
        
        # Save combined dataset
        combined_data.to_csv('B-Bank_Train_data.csv', index=False)
        print("✓ B-Bank combined dataset saved")
        
        # Train B-Bank model (enhanced version)
        return train_bbank_enhanced_model(combined_data)
        
    except Exception as e:
        print(f"✗ Error creating B-Bank model: {e}")
        return False

def train_bbank_enhanced_model(data):
    """Train enhanced B-Bank model with Source_Bank feature"""
    print(f"🔄 Training enhanced B-Bank model...")
    
    # Data preprocessing (same as other models but with Source_Bank)
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                    'Amount_invested_monthly', 'Monthly_Balance']
    
    # Convert to numeric
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Process credit history
    if 'Credit_History_Age' in data.columns:
        data['Credit_History_Age_Years'] = data['Credit_History_Age'].apply(extract_credit_history_age)
    
    # Binary conversion
    if 'Payment_of_Min_Amount' in data.columns:
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})
    
    # Risk level mapping
    if 'Credit_Mix' in data.columns:
        risk_mapping = {'Good': 0, 'Standard': 1, 'Bad': 2}
        data['Risk_Level'] = data['Credit_Mix'].map(risk_mapping)
        data['Risk_Level'] = data['Risk_Level'].fillna(1)
    
    # Feature selection (including Source_Bank)
    numeric_features = [col for col in numeric_cols + ['Credit_History_Age_Years', 'Payment_of_Min_Amount'] 
                       if col in data.columns]
    categorical_cols = ['Occupation', 'Source_Bank']
    
    # Handle missing values
    data = data.dropna(subset=['Risk_Level'])
    
    # Fill missing values
    for col in numeric_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown')
    
    # Prepare features
    available_features = [col for col in numeric_features + categorical_cols if col in data.columns]
    X = data[available_features]
    y = data['Risk_Level']
    
    # Enhanced preprocessing
    numeric_available = [col for col in numeric_features if col in data.columns]
    categorical_available = [col for col in categorical_cols if col in data.columns]
    
    transformers = []
    if numeric_available:
        transformers.append(('num', RobustScaler(), numeric_available))
    if categorical_available:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_available))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Enhanced B-Bank model
    base_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,  # More trees for B-Bank
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train base model
    base_model.fit(X_train, y_train)
    
    # Calibrate for better probabilities
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=min(3, len(np.unique(y_train))))
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ B-Bank Enhanced Model trained - Accuracy: {accuracy:.4f}")
    
    # Save model and metadata
    model_path = os.path.join(MODEL_DIR, 'B-Bank_loan_risk_model.pkl')
    joblib.dump(calibrated_model, model_path)
    
    # Save metadata
    metadata = {
        'accuracy': accuracy,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_names': available_features,
        'trained_at': datetime.now().isoformat(),
        'model_name': 'B-Bank Enhanced',
        'source_banks': data['Source_Bank'].value_counts().to_dict()
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    
    print(f"✓ B-Bank enhanced model saved to {model_path}")
    return True

def main():
    """Main function to train all models"""
    print("🚀 Training Individual Bank Models")
    print("=" * 50)
    
    # Train individual bank models
    models_to_train = [
        ('SB_Train_data.csv', 'StandardBank', 'SB_loan_risk_model.pkl'),
        ('PB_Train_data.csv', 'PostBank', 'PB_loan_risk_model.pkl'),
        ('FNB_Train_data.csv', 'FNB', 'FNB_loan_risk_model.pkl')
    ]
    
    success_count = 0
    for csv_file, model_name, output_file in models_to_train:
        if train_bank_model(csv_file, model_name, output_file):
            success_count += 1
    
    # Create B-Bank combined model
    if create_bbank_combined_model():
        success_count += 1
    
    print(f"\n🎉 Training completed: {success_count}/4 models trained successfully")
    print(f"📁 Models saved in: {MODEL_DIR}/")
    
    # List saved models
    print("\n📋 Saved Models:")
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith('.pkl') and not filename.endswith('_metadata.pkl'):
            filepath = os.path.join(MODEL_DIR, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  • {filename} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    main()
