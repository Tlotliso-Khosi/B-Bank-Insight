import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import seaborn as sns

# Load the datasets
sb_data = pd.read_csv('SB_Train_data.csv')
pb_data = pd.read_csv('PB_Train_data.csv')

print("=== DATASET COMPARISON ANALYSIS ===")
print(f"SB Dataset shape: {sb_data.shape}")
print(f"PB Dataset shape: {pb_data.shape}")

# 1. ANALYZE RISK DISTRIBUTION DIFFERENCES
print("\n1. RISK DISTRIBUTION ANALYSIS")
print("-" * 40)

# Process Credit_Mix for both datasets
def analyze_credit_mix(data, dataset_name):
    if 'Credit_Mix' in data.columns:
        credit_mix_counts = data['Credit_Mix'].value_counts()
        credit_mix_pct = data['Credit_Mix'].value_counts(normalize=True) * 100
        
        print(f"\n{dataset_name} Credit Mix Distribution:")
        for category in ['Good', 'Standard', 'Bad']:
            count = credit_mix_counts.get(category, 0)
            pct = credit_mix_pct.get(category, 0)
            print(f"  {category}: {count} ({pct:.1f}%)")
        
        # Check for missing values
        missing_count = data['Credit_Mix'].isna().sum()
        missing_pct = (missing_count / len(data)) * 100
        print(f"  Missing/Unknown: {missing_count} ({missing_pct:.1f}%)")
        
        return credit_mix_pct
    else:
        print(f"{dataset_name}: No Credit_Mix column found")
        return None

sb_risk_dist = analyze_credit_mix(sb_data, "SB Dataset")
pb_risk_dist = analyze_credit_mix(pb_data, "PB Dataset")

# 2. ANALYZE FEATURE DISTRIBUTIONS
print("\n2. FEATURE DISTRIBUTION ANALYSIS")
print("-" * 40)

numeric_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 
                   'Credit_Utilization_Ratio', 'Num_of_Delayed_Payment']

def compare_feature_distributions(sb_data, pb_data, features):
    comparison_stats = []
    
    for feature in features:
        if feature in sb_data.columns and feature in pb_data.columns:
            # Convert to numeric
            sb_values = pd.to_numeric(sb_data[feature], errors='coerce')
            pb_values = pd.to_numeric(pb_data[feature], errors='coerce')
            
            # Calculate statistics
            sb_stats = {
                'mean': sb_values.mean(),
                'median': sb_values.median(),
                'std': sb_values.std(),
                'min': sb_values.min(),
                'max': sb_values.max()
            }
            
            pb_stats = {
                'mean': pb_values.mean(),
                'median': pb_values.median(),
                'std': pb_values.std(),
                'min': pb_values.min(),
                'max': pb_values.max()
            }
            
            # Calculate differences
            mean_diff = abs(sb_stats['mean'] - pb_stats['mean']) / max(sb_stats['mean'], pb_stats['mean']) * 100
            
            comparison_stats.append({
                'Feature': feature,
                'SB_Mean': sb_stats['mean'],
                'PB_Mean': pb_stats['mean'],
                'Mean_Diff_%': mean_diff,
                'SB_Std': sb_stats['std'],
                'PB_Std': pb_stats['std']
            })
            
            print(f"\n{feature}:")
            print(f"  SB: Mean={sb_stats['mean']:.2f}, Std={sb_stats['std']:.2f}")
            print(f"  PB: Mean={pb_stats['mean']:.2f}, Std={pb_stats['std']:.2f}")
            print(f"  Mean Difference: {mean_diff:.1f}%")
    
    return pd.DataFrame(comparison_stats)

feature_comparison = compare_feature_distributions(sb_data, pb_data, numeric_features)

# 3. ANALYZE DATA QUALITY
print("\n3. DATA QUALITY ANALYSIS")
print("-" * 40)

def analyze_data_quality(data, dataset_name):
    print(f"\n{dataset_name} Data Quality:")
    
    # Missing values
    missing_pct = (data.isnull().sum() / len(data) * 100).round(2)
    high_missing = missing_pct[missing_pct > 10]
    
    if len(high_missing) > 0:
        print("  High missing value columns (>10%):")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct}%")
    else:
        print("  No columns with high missing values (>10%)")
    
    # Duplicate records
    duplicates = data.duplicated().sum()
    print(f"  Duplicate records: {duplicates}")
    
    # Unique customers
    if 'Customer_ID' in data.columns:
        unique_customers = data['Customer_ID'].nunique()
        total_records = len(data)
        print(f"  Unique customers: {unique_customers}")
        print(f"  Records per customer: {total_records/unique_customers:.1f}")

analyze_data_quality(sb_data, "SB Dataset")
analyze_data_quality(pb_data, "PB Dataset")

# 4. IDENTIFY POTENTIAL CAUSES OF MODEL DIFFERENCES
print("\n4. POTENTIAL CAUSES OF MODEL DIFFERENCES")
print("-" * 50)

causes = []

# Check risk distribution differences
if sb_risk_dist is not None and pb_risk_dist is not None:
    good_diff = abs(sb_risk_dist.get('Good', 0) - pb_risk_dist.get('Good', 0))
    bad_diff = abs(sb_risk_dist.get('Bad', 0) - pb_risk_dist.get('Bad', 0))
    
    if good_diff > 20 or bad_diff > 20:
        causes.append("MAJOR: Significant difference in risk class distributions")
        print(f"• Risk distribution difference: Good={good_diff:.1f}%, Bad={bad_diff:.1f}%")

# Check feature distribution differences
high_diff_features = feature_comparison[feature_comparison['Mean_Diff_%'] > 50]
if len(high_diff_features) > 0:
    causes.append("MAJOR: Large differences in feature distributions")
    print("• Features with >50% mean difference:")
    for _, row in high_diff_features.iterrows():
        print(f"  - {row['Feature']}: {row['Mean_Diff_%']:.1f}% difference")

# Check data size differences
size_ratio = max(len(sb_data), len(pb_data)) / min(len(sb_data), len(pb_data))
if size_ratio > 2:
    causes.append("MODERATE: Significant dataset size difference")
    print(f"• Dataset size ratio: {size_ratio:.1f}:1")

# Check missing data patterns
sb_missing_avg = sb_data.isnull().sum().mean()
pb_missing_avg = pb_data.isnull().sum().mean()
missing_diff = abs(sb_missing_avg - pb_missing_avg)

if missing_diff > 100:
    causes.append("MODERATE: Different missing data patterns")
    print(f"• Missing data difference: {missing_diff:.0f} average missing values per column")

if not causes:
    print("• No major obvious causes detected in basic analysis")

print(f"\nSUMMARY: Found {len(causes)} potential causes for model differences")
for i, cause in enumerate(causes, 1):
    print(f"{i}. {cause}")