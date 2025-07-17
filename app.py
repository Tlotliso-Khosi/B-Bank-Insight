from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, render_template_string
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import warnings
from functools import wraps
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production!

# Model file paths
MODEL_FILES = {
    'sb': 'SB_loan_risk_model.pkl',
    'pb': 'PB_loan_risk_model.pkl',
    'fnb': 'FNB_loan_risk_model.pkl',
    'bbank': 'B-Bank_loan_risk_model.pkl'
}

# Demo user database (replace with real database in production)
DEMO_USERS = {
    'john.doe@standardbank.com': {
        'password': 'password123',
        'name': 'John Doe',
        'bank': 'SB',
        'employee_id': 'SB001',
        'role': 'Loan Officer'
    },
    'jane.smith@premierbank.com': {
        'password': 'password123',
        'name': 'Jane Smith',
        'bank': 'PB',
        'employee_id': 'PB001',
        'role': 'Risk Analyst'
    },
    'mike.johnson@fnb.com': {
        'password': 'password123',
        'name': 'Mike Johnson',
        'bank': 'FNB',
        'employee_id': 'FNB001',
        'role': 'Credit Manager'
    },
    'admin@bbank.com': {
        'password': 'admin123',
        'name': 'B-Bank Admin',
        'bank': 'B-Bank',
        'employee_id': 'BB001',
        'role': 'System Administrator'
    }
}

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/auth/login', methods=['POST'])
def auth_login():
    try:
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        bank = request.form.get('bank', '')
        
        # Validate input
        if not email or not password or not bank:
            return jsonify({
                'success': False,
                'message': 'Please fill in all required fields.'
            })
        
        # Check if user exists
        if email not in DEMO_USERS:
            return jsonify({
                'success': False,
                'message': 'Invalid email address. Please check your credentials.'
            })
        
        user = DEMO_USERS[email]
        
        # Verify password
        if user['password'] != password:
            return jsonify({
                'success': False,
                'message': 'Invalid password. Please check your credentials.'
            })
        
        # Verify bank affiliation
        if user['bank'] != bank:
            return jsonify({
                'success': False,
                'message': f'Bank affiliation mismatch. This account is registered with {user["bank"]}.'
            })
        
        # Login successful - create session
        session['user_email'] = email
        session['user_name'] = user['name']
        session['user_bank'] = user['bank']
        session['user_role'] = user['role']
        session['employee_id'] = user['employee_id']
        
        return jsonify({
            'success': True,
            'message': 'Login successful!',
            'user': {
                'name': user['name'],
                'bank': user['bank'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Login error: {str(e)}'
        })

@app.route('/auth/signup', methods=['POST'])
def auth_signup():
    try:
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        name = request.form.get('name', '').strip()
        employee_id = request.form.get('employee_id', '').strip()
        bank = request.form.get('bank', '')
        
        # Validate input
        if not all([email, password, confirm_password, name, employee_id, bank]):
            return jsonify({
                'success': False,
                'message': 'Please fill in all required fields.'
            })
        
        # Check if user already exists
        if email in DEMO_USERS:
            return jsonify({
                'success': False,
                'message': 'An account with this email already exists.'
            })
        
        # Validate password match
        if password != confirm_password:
            return jsonify({
                'success': False,
                'message': 'Passwords do not match.'
            })
        
        # Validate password strength
        if len(password) < 8:
            return jsonify({
                'success': False,
                'message': 'Password must be at least 8 characters long.'
            })
        
        # Validate email domain (basic check)
        domain = email.split('@')[1] if '@' in email else ''
        valid_domains = {
            'SB': ['standardbank.com', 'sb.co.za'],
            'PB': ['premierbank.com', 'pb.co.za'],
            'FNB': ['fnb.com', 'fnb.co.za']
        }
        
        if bank in valid_domains:
            if not any(d in domain for d in valid_domains[bank]):
                return jsonify({
                    'success': False,
                    'message': f'Please use your official {bank} email address.'
                })
        
        # Create new user (in production, save to database)
        DEMO_USERS[email] = {
            'password': password,
            'name': name,
            'bank': bank,
            'employee_id': employee_id,
            'role': 'Loan Officer'  # Default role
        }
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully! You can now login.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Signup error: {str(e)}'
        })

@app.route('/dashboard')
@login_required
def dashboard():
    # This renders the main risk assessment system
    return render_template_string(HTML_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

# Enhanced HTML template with detailed risk analysis
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Multi-Model Risk Assessment System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 1px solid #e1e4e8;
        }

        header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .section {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin-bottom: 30px;
        }

        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e1e4e8;
        }

        /* B-Bank Primary Decision Section */
        .primary-decision {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: 3px solid #5a67d8;
            position: relative;
            overflow: hidden;
        }

        .primary-decision::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 10px,
                rgba(255,255,255,0.05) 10px,
                rgba(255,255,255,0.05) 20px
            );
            animation: shimmer 3s linear infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }

        .primary-decision h2 {
            color: white;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            position: relative;
            z-index: 1;
        }

        .primary-decision .content {
            position: relative;
            z-index: 1;
        }

        .primary-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background-color: #ffd700;
            color: #333;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            z-index: 2;
        }

        /* Form Styles */
        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .form-group input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }

        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background-color 0.3s;
        }

        .btn:hover {
            background-color: #2980b9;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 30px;
            font-size: 18px;
        }

        /* Processing */
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        #processing-section {
            text-align: center;
        }

        /* Customer Details */
        .customer-details {
            margin-bottom: 30px;
        }

        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }

        .detail-item {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }

        .label {
            font-weight: 600;
            margin-right: 5px;
            color: #2c3e50;
        }

        /* Risk Analysis Grid */
        .risk-analysis-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .supporting-models {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .model-result {
            padding: 20px;
            border-radius: 8px;
            position: relative;
            border: 2px solid #e1e4e8;
        }

        .model-result.sb-model {
            background-color: #e8f5e8;
            border-color: #27ae60;
        }

        .model-result.pb-model {
            background-color: #f0f8ff;
            border-color: #3498db;
        }

        .model-result.fnb-model {
            background-color: #fff5f5;
            border-color: #e74c3c;
        }

        .model-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            background-color: #34495e;
            color: white;
        }

        /* Risk Meters */
        .risk-level {
            margin: 20px 0;
            text-align: center;
        }

        .risk-meter {
            height: 25px;
            background-color: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;
            position: relative;
        }

        .risk-fill {
            height: 100%;
            width: 0;
            transition: width 1.5s ease-in-out;
            position: relative;
        }

        .risk-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: progress-shine 2s ease-in-out infinite;
        }

        @keyframes progress-shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .low-risk {
            background: linear-gradient(90deg, #2ecc71, #27ae60);
        }

        .medium-risk {
            background: linear-gradient(90deg, #f39c12, #e67e22);
        }

        .high-risk {
            background: linear-gradient(90deg, #e74c3c, #c0392b);
        }

        /* Detailed Analysis */
        .detailed-analysis {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .analysis-section {
            margin-bottom: 25px;
        }

        .analysis-section h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }

        .factor-list {
            list-style: none;
            padding: 0;
        }

        .factor-item {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }

        .factor-positive {
            background-color: #d5f4e6;
            border-left-color: #27ae60;
        }

        .factor-negative {
            background-color: #ffeaa7;
            border-left-color: #f39c12;
        }

        .factor-critical {
            background-color: #fab1a0;
            border-left-color: #e74c3c;
        }

        /* Enhanced Recommendation Box Styling */
        .recommendation-box {
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 0;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            overflow: hidden;
        }

        .recommendation-header {
            background: rgba(255,255,255,0.1);
            padding: 20px 25px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }

        .recommendation-header h4 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .recommendation-content {
            padding: 0;
        }

        .recommendation-section {
            padding: 20px 25px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .recommendation-section:last-child {
            border-bottom: none;
        }

        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #fff;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }

        .info-item {
            background: rgba(255,255,255,0.1);
            padding: 12px 15px;
            border-radius: 8px;
            border-left: 4px solid rgba(255,255,255,0.3);
        }

        .info-label {
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .info-value {
            font-size: 16px;
            font-weight: 600;
        }

        .risk-factors-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .risk-factor-item {
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 4px solid #ff6b6b;
            font-size: 14px;
        }

        .strengths-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .strength-item {
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 4px solid #51cf66;
            font-size: 14px;
        }

        .terms-package {
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
        }

        .package-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
        }

        .terms-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }

        .term-item {
            background: rgba(255,255,255,0.1);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }

        .term-label {
            font-size: 11px;
            opacity: 0.8;
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .term-value {
            font-size: 14px;
            font-weight: 600;
        }

        .action-badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }

        .approval-badge {
            background: linear-gradient(45deg, #51cf66, #40c057);
        }

        .conditional-badge {
            background: linear-gradient(45deg, #ffd43b, #fab005);
            color: #333;
        }

        .review-badge {
            background: linear-gradient(45deg, #ff8cc8, #ff6b9d);
        }

        .decline-badge {
            background: linear-gradient(45deg, #ff6b6b, #fa5252);
        }

        .improvement-path {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }

        .path-timeline {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .timeline-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #74b9ff;
        }

        .timeline-title {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .timeline-content {
            font-size: 13px;
            opacity: 0.9;
            line-height: 1.4;
        }

        /* Model Comparison */
        .model-comparison {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .comparison-item {
            text-align: center;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        .hidden {
            display: none;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .risk-analysis-grid {
                grid-template-columns: 1fr;
            }
            
            .supporting-models {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .details-grid {
                grid-template-columns: 1fr;
            }
            
            .comparison-grid {
                grid-template-columns: 1fr;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }

            .terms-grid {
                grid-template-columns: 1fr;
            }

            .path-timeline {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏦 Advanced Multi-Model Risk Assessment System</h1>
            <p>Comprehensive loan risk analysis powered by B-Bank's master algorithm</p>
        </header>

        <!-- Level 1: Input Section -->
        <section id="input-section" class="section">
            <h2>📋 Level 1: Customer Information Input</h2>
            <form id="loan-form">
                <div class="form-group">
                    <label for="customer-id">Customer ID:</label>
                    <input type="text" id="customer-id" name="customer_id" placeholder="e.g., CUS_0x16ae" required>
                </div>
                <div class="form-group">
                    <label for="loan-amount">Loan Amount ($):</label>
                    <input type="number" id="loan-amount" name="loan_amount" min="100" step="100" placeholder="e.g., 50000" required>
                </div>
                <button type="submit" class="btn btn-primary">🔍 Analyze with All Models</button>
            </form>
        </section>

        <!-- Level 2: Processing -->
        <section id="processing-section" class="section hidden">
            <h2>⚙️ Level 2: Multi-Model Processing</h2>
            <div class="loader"></div>
            <p>🔍 Retrieving customer data across all banking systems...</p>
            <p>📊 Loading saved models...</p>
            <p>🧮 Calculating normalized financial metrics...</p>
            <p>🤖 Processing through SB, PB, and FNB models...</p>
            <p>🏆 Running B-Bank master algorithm for final decision...</p>
            <p>📈 Generating detailed risk analysis...</p>
        </section>

        <!-- Level 3: Output Section -->
        <section id="output-section" class="section hidden">
            <h2>📈 Level 3: Comprehensive Risk Assessment Results</h2>
            
            <!-- Customer Profile -->
            <div class="customer-details">
                <h3>👤 Customer Profile</h3>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="label">Name:</span>
                        <span id="customer-name" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Age:</span>
                        <span id="customer-age" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Occupation:</span>
                        <span id="customer-occupation" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Annual Income:</span>
                        <span id="customer-income" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Current Outstanding Debt:</span>
                        <span id="customer-debt" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Requested Loan:</span>
                        <span id="requested-loan" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Total Debt After Loan:</span>
                        <span id="total-debt" class="value">-</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Debt-to-Income Ratio:</span>
                        <span id="debt-income-ratio" class="value">-</span>
                    </div>
                </div>
            </div>

            <!-- Risk Analysis Grid -->
            <div class="risk-analysis-grid">
                <!-- B-Bank Primary Decision -->
                <div class="primary-decision section">
                    <div class="primary-badge">PRIMARY DECISION</div>
                    <div class="content">
                        <h2>🏆 B-Bank Master Algorithm Decision</h2>
                        <div class="risk-level">
                            <div class="risk-meter">
                                <div class="risk-fill" id="bbank-risk-fill"></div>
                            </div>
                            <h3 id="bbank-risk-level" style="color: white; margin: 15px 0;">-</h3>
                            <p id="bbank-risk-percentage" style="font-size: 24px; font-weight: bold;">-</p>
                        </div>
                        <p id="bbank-risk-description" style="margin: 15px 0; font-size: 16px;">-</p>
                        <p style="margin: 10px 0;"><strong>Approval Probability:</strong> <span id="bbank-approval-chance">-</span></p>
                        <p style="margin: 10px 0;"><strong>Confidence Level:</strong> <span id="bbank-confidence">-</span></p>
                    </div>
                </div>

                <!-- Supporting Models -->
                <div class="supporting-models">
                    <div class="model-result sb-model">
                        <div class="model-badge">SB Bank</div>
                        <h4>SB Model Assessment</h4>
                        <div class="risk-level">
                            <div class="risk-meter">
                                <div class="risk-fill" id="sb-risk-fill"></div>
                            </div>
                            <h5 id="sb-risk-level">-</h5>
                            <p id="sb-risk-percentage">-</p>
                        </div>
                        <p id="sb-approval-chance">-</p>
                    </div>

                    <div class="model-result pb-model">
                        <div class="model-badge">PB Bank</div>
                        <h4>PB Model Assessment</h4>
                        <div class="risk-level">
                            <div class="risk-meter">
                                <div class="risk-fill" id="pb-risk-fill"></div>
                            </div>
                            <h5 id="pb-risk-level">-</h5>
                            <p id="pb-risk-percentage">-</p>
                        </div>
                        <p id="pb-approval-chance">-</p>
                    </div>

                    <div class="model-result fnb-model">
                        <div class="model-badge">FNB Bank</div>
                        <h4>FNB Model Assessment</h4>
                        <div class="risk-level">
                            <div class="risk-meter">
                                <div class="risk-fill" id="fnb-risk-fill"></div>
                            </div>
                            <h5 id="fnb-risk-level">-</h5>
                            <p id="fnb-risk-percentage">-</p>
                        </div>
                        <p id="fnb-approval-chance">-</p>
                    </div>
                </div>
            </div>

            <!-- Model Comparison -->
            <div class="model-comparison">
                <h3>📊 Multi-Model Analysis Comparison</h3>
                <p id="model-agreement">-</p>
                <div class="comparison-grid">
                    <div class="comparison-item">
                        <h5>SB Bank</h5>
                        <p id="sb-comparison">-</p>
                    </div>
                    <div class="comparison-item">
                        <h5>PB Bank</h5>
                        <p id="pb-comparison">-</p>
                    </div>
                    <div class="comparison-item">
                        <h5>FNB Bank</h5>
                        <p id="fnb-comparison">-</p>
                    </div>
                    <div class="comparison-item">
                        <h5>B-Bank Master</h5>
                        <p id="bbank-comparison">-</p>
                    </div>
                </div>
            </div>

            <!-- Detailed Risk Analysis -->
            <div class="detailed-analysis">
                <h3>🔍 Detailed Risk Analysis & Decision Factors</h3>
                
                <div class="analysis-section">
                    <h4>✅ Positive Risk Factors</h4>
                    <ul class="factor-list" id="positive-factors">
                        <!-- Populated by JavaScript -->
                    </ul>
                </div>

                <div class="analysis-section">
                    <h4>⚠️ Risk Concerns</h4>
                    <ul class="factor-list" id="risk-concerns">
                        <!-- Populated by JavaScript -->
                    </ul>
                </div>

                <div class="analysis-section">
                    <h4>🚨 Critical Risk Factors</h4>
                    <ul class="factor-list" id="critical-factors">
                        <!-- Populated by JavaScript -->
                    </ul>
                </div>

                <div class="analysis-section">
                    <h4>💰 Financial Health Indicators</h4>
                    <ul class="factor-list" id="financial-indicators">
                        <!-- Populated by JavaScript -->
                    </ul>
                </div>
            </div>

            <!-- Final Recommendation -->
            <div class="recommendation-box">
                <div class="recommendation-header">
                    <h4>🎯 Final B-Bank Recommendation</h4>
                </div>
                <div class="recommendation-content" id="final-recommendation-content">
                    <!-- This will be populated by JavaScript with structured HTML -->
                </div>
            </div>
            
            <button id="new-assessment" class="btn">🔄 New Assessment</button>
        </section>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const loanForm = document.getElementById('loan-form');
            const inputSection = document.getElementById('input-section');
            const processingSection = document.getElementById('processing-section');
            const outputSection = document.getElementById('output-section');
            const newAssessmentBtn = document.getElementById('new-assessment');
            
            loanForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                inputSection.classList.add('hidden');
                processingSection.classList.remove('hidden');
                outputSection.classList.add('hidden');
                
                const formData = new FormData(loanForm);
                
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    processingSection.classList.add('hidden');
                    
                    if (data.success) {
                        updateResults(data);
                        outputSection.classList.remove('hidden');
                    } else {
                        alert(data.message);
                        inputSection.classList.remove('hidden');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred. Please try again.');
                    inputSection.classList.remove('hidden');
                    processingSection.classList.add('hidden');
                });
            });
            
            newAssessmentBtn.addEventListener('click', function() {
                loanForm.reset();
                inputSection.classList.remove('hidden');
                outputSection.classList.add('hidden');
            });
            
            function updateResults(data) {
                // Update customer details
                document.getElementById('customer-name').textContent = data.customer.name;
                document.getElementById('customer-age').textContent = data.customer.age;
                document.getElementById('customer-occupation').textContent = data.customer.occupation;
                document.getElementById('customer-income').textContent = data.customer.annual_income;
                document.getElementById('customer-debt').textContent = data.customer.original_debt;
                document.getElementById('requested-loan').textContent = data.customer.requested_loan;
                document.getElementById('total-debt').textContent = data.customer.total_debt;
                document.getElementById('debt-income-ratio').textContent = data.customer.debt_income_ratio;
                
                // Update B-Bank primary decision
                updateModelResult('bbank', data.bbank_risk);
                document.getElementById('bbank-confidence').textContent = data.bbank_confidence.level;
                
                // Update supporting models
                updateModelResult('sb', data.sb_risk);
                updateModelResult('pb', data.pb_risk);
                updateModelResult('fnb', data.fnb_risk);
                
                // Update model comparison
                updateModelComparison(data);
                
                // Update detailed analysis
                updateDetailedAnalysis(data.detailed_analysis);
                
                // Update final recommendation with structured HTML
                updateFinalRecommendation(data);
            }
            
            function updateModelResult(prefix, riskData) {
                document.getElementById(prefix + '-risk-level').textContent = riskData.risk_level;
                document.getElementById(prefix + '-risk-percentage').textContent = riskData.risk_percentage;
                if (prefix !== 'bbank') {
                    document.getElementById(prefix + '-approval-chance').textContent = riskData.approval_chance;
                } else {
                    document.getElementById(prefix + '-risk-description').textContent = riskData.risk_description;
                    document.getElementById(prefix + '-approval-chance').textContent = riskData.approval_chance;
                }
                
                setRiskMeter(prefix + '-risk-fill', riskData.risk_level, riskData.risk_percentage);
            }
            
            function updateModelComparison(data) {
                document.getElementById('model-agreement').textContent = data.model_agreement;
                document.getElementById('sb-comparison').textContent = data.sb_risk.risk_percentage;
                document.getElementById('pb-comparison').textContent = data.pb_risk.risk_percentage;
                document.getElementById('fnb-comparison').textContent = data.fnb_risk.risk_percentage;
                document.getElementById('bbank-comparison').textContent = data.bbank_risk.risk_percentage;
            }
            
            function updateDetailedAnalysis(analysis) {
                updateFactorList('positive-factors', analysis.positive_factors, 'factor-positive');
                updateFactorList('risk-concerns', analysis.risk_concerns, 'factor-negative');
                updateFactorList('critical-factors', analysis.critical_factors, 'factor-critical');
                updateFactorList('financial-indicators', analysis.financial_indicators, 'factor-positive');
            }
            
            function updateFactorList(elementId, factors, className) {
                const list = document.getElementById(elementId);
                list.innerHTML = '';
                factors.forEach(factor => {
                    const li = document.createElement('li');
                    li.className = 'factor-item ' + className;
                    li.textContent = factor;
                    list.appendChild(li);
                });
            }
            
            function updateFinalRecommendation(data) {
                const container = document.getElementById('final-recommendation-content');
                container.innerHTML = data.final_recommendation_html;
            }
            
            function setRiskMeter(elementId, riskLevel, riskPercentage) {
                const riskFill = document.getElementById(elementId);
                const percentage = parseFloat(riskPercentage);
                
                riskFill.classList.remove('low-risk', 'medium-risk', 'high-risk');
                
                if (riskLevel === 'Low Risk') {
                    riskFill.classList.add('low-risk');
                } else if (riskLevel === 'Medium Risk') {
                    riskFill.classList.add('medium-risk');
                } else {
                    riskFill.classList.add('high-risk');
                }
                
                setTimeout(() => {
                    riskFill.style.width = percentage + '%';
                }, 500);
            }
        });
    </script>
</body>
</html>
'''

# Load datasets from local CSV files
def load_all_datasets():
    datasets = {}
    
    # Load SB dataset
    try:
        datasets['sb'] = pd.read_csv('SB_Train_data.csv')
        print(f"✓ SB Dataset loaded: {datasets['sb'].shape}")
    except Exception as e:
        print(f"✗ Error loading SB dataset: {e}")
        datasets['sb'] = pd.DataFrame()
    
    # Load PB dataset
    try:
        datasets['pb'] = pd.read_csv('PB_Train_data.csv')
        print(f"✓ PB Dataset loaded: {datasets['pb'].shape}")
    except Exception as e:
        print(f"✗ Error loading PB dataset: {e}")
        datasets['pb'] = pd.DataFrame()
    
    # Load FNB dataset
    try:
        datasets['fnb'] = pd.read_csv('FNB_Train_data.csv')
        print(f"✓ FNB Dataset loaded: {datasets['fnb'].shape}")
    except Exception as e:
        print(f"✗ Error loading FNB dataset: {e}")
        datasets['fnb'] = pd.DataFrame()
    
    # Load B-Bank dataset
    try:
        datasets['bbank'] = pd.read_csv('B-Bank_Train_data.csv')
        print(f"✓ B-Bank Dataset loaded: {datasets['bbank'].shape}")
    except Exception as e:
        print(f"✗ Error loading B-Bank dataset: {e}")
        # If B-Bank dataset doesn't exist, create it from other datasets
        try:
            all_data = []
            for name, data in datasets.items():
                if not data.empty and len(data) > 0:
                    data_copy = data.copy()
                    data_copy['Source_Bank'] = name.upper()
                    all_data.append(data_copy)
            
            if all_data:
                datasets['bbank'] = pd.concat(all_data, ignore_index=True)
                # Remove duplicates based on Customer_ID if present
                if 'Customer_ID' in datasets['bbank'].columns:
                    datasets['bbank'] = datasets['bbank'].drop_duplicates(subset=['Customer_ID'], keep='first')
                print(f"✓ B-Bank Combined Dataset created: {datasets['bbank'].shape}")
            else:
                print("✗ No data available for B-Bank dataset")
                datasets['bbank'] = pd.DataFrame()
        except Exception as e2:
            print(f"✗ Error creating B-Bank dataset: {e2}")
            datasets['bbank'] = pd.DataFrame()
    
    return datasets

# Enhanced model training
def train_enhanced_model(data, model_name):
    if data.empty or len(data) == 0:
        print(f"✗ No data available for {model_name} model")
        return None, None
    
    print(f"🔄 Training {model_name} model with {len(data)} samples...")
    
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
    
    # Enhanced credit history processing
    def extract_credit_history_age(age_str):
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
    
    if 'Credit_History_Age' in data.columns:
        data['Credit_History_Age_Years'] = data['Credit_History_Age'].apply(extract_credit_history_age)
    
    # Binary conversion
    if 'Payment_of_Min_Amount' in data.columns:
        data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0})
    
    # Risk level mapping with better handling
    if 'Credit_Mix' in data.columns:
        risk_mapping = {'Good': 0, 'Standard': 1, 'Bad': 2}
        data['Risk_Level'] = data['Credit_Mix'].map(risk_mapping)
        # Fill missing values based on other indicators
        data['Risk_Level'] = data['Risk_Level'].fillna(1)
    else:
        print(f"⚠️ Warning: No Credit_Mix column found in {model_name} dataset")
        return None, None
    
    # Feature selection
    numeric_features = [col for col in numeric_cols + ['Credit_History_Age_Years', 'Payment_of_Min_Amount'] 
                       if col in data.columns]
    categorical_cols = ['Occupation'] if 'Occupation' in data.columns else []
    
    # For B-Bank, add Source_Bank as a feature if it exists
    if model_name == 'B-Bank' and 'Source_Bank' in data.columns:
        categorical_cols.append('Source_Bank')
    
    # Handle missing values
    data = data.dropna(subset=['Risk_Level'])
    if len(data) < 50:
        print(f"✗ Insufficient data for {model_name} model (only {len(data)} samples)")
        return None, None
    
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
    
    # Check if we have enough variety in target variable
    if len(y.unique()) < 2:
        print(f"✗ Insufficient target variety for {model_name} model")
        return None, None
    
    # Enhanced preprocessing
    numeric_available = [col for col in numeric_features if col in data.columns]
    categorical_available = [col for col in categorical_cols if col in data.columns]
    
    transformers = []
    if numeric_available:
        transformers.append(('num', RobustScaler(), numeric_available))
    if categorical_available:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_available))
    
    if not transformers:
        print(f"✗ No valid features found for {model_name} model")
        return None, None
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Enhanced model for B-Bank
    if model_name == 'B-Bank':
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
    else:
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
    
    try:
        # Check if we have enough data for train/test split
        if len(X) < 10:
            print(f"✗ Too few samples for {model_name} model training")
            return None, None
        
        # Use stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            # If stratification fails, use regular split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train base model
        base_model.fit(X_train, y_train)
        
        # Calibrate for better probabilities
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=min(3, len(np.unique(y_train))))
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = calibrated_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ {model_name} Model trained successfully - Accuracy: {accuracy:.4f}")
        
        # Save the model
        model_file = MODEL_FILES[model_name.lower().replace('-', '')]
        joblib.dump(calibrated_model, model_file)
        print(f"💾 {model_name} Model saved to {model_file}")
        
        # Get feature importance
        feature_importance = None
        if hasattr(base_model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = base_model.named_steps['classifier'].feature_importances_
        
        return calibrated_model, {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'feature_names': available_features
        }
        
    except Exception as e:
        print(f"✗ Error training {model_name} model: {e}")
        return None, None

def load_or_train_models():
    """Load existing models or train new ones if they don't exist"""
    print("🚀 Loading datasets and initializing models...")
    datasets = load_all_datasets()
    
    models = {}
    model_stats = {}
    
    for name, data in datasets.items():
        model_name = name.upper()
        if name == 'bbank':
            model_name = 'B-Bank'
        
        model_file = MODEL_FILES[name]
        
        # Check if model file exists
        if os.path.exists(model_file):
            try:
                # Load existing model
                models[name] = joblib.load(model_file)
                model_stats[name] = {'accuracy': 'Loaded from file'}
                print(f"📁 {model_name} Model loaded from {model_file}")
            except Exception as e:
                print(f"✗ Error loading {model_name} model from {model_file}: {e}")
                print(f"🔄 Training new {model_name} model...")
                models[name], model_stats[name] = train_enhanced_model(data, model_name)
        else:
            # Train new model
            print(f"🔄 No saved model found for {model_name}, training new model...")
            models[name], model_stats[name] = train_enhanced_model(data, model_name)
    
    return models, model_stats, datasets

# Load data and initialize models
models, model_stats, datasets = load_or_train_models()

# Risk levels
risk_levels = {
    0: {'name': 'Low Risk', 'description': 'Excellent credit profile with minimal default risk', 'approval_chance': '90-100%'},
    1: {'name': 'Medium Risk', 'description': 'Acceptable credit profile with moderate risk factors', 'approval_chance': '60-85%'},
    2: {'name': 'High Risk', 'description': 'Poor credit profile with significant default risk', 'approval_chance': '10-50%'}
}

def get_customer_data(customer_id):
    """Find customer in any dataset"""
    for name, data in datasets.items():
        if not data.empty:
            customer = data[data['Customer_ID'] == customer_id]
            if not customer.empty:
                return customer, name.upper()
    return None, None

def calculate_enhanced_averages(customer_data, loan_amount):
    """Calculate enhanced averages with additional metrics"""
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                    'Amount_invested_monthly', 'Monthly_Balance']
    
    averages = {}
    for col in numeric_cols:
        if col in customer_data.columns:
            try:
                numeric_values = pd.to_numeric(customer_data[col], errors='coerce')
                averages[col] = numeric_values.mean() if not numeric_values.isna().all() else 0
            except:
                averages[col] = 0
    
    # Enhanced credit history
    if 'Credit_History_Age' in customer_data.columns:
        credit_history = []
        for age_str in customer_data['Credit_History_Age']:
            try:
                if pd.isna(age_str) or age_str == 'NA':
                    continue
                parts = str(age_str).split()
                years = 0
                months = 0
                if 'Years' in str(age_str):
                    years = float(parts[0])
                if 'Months' in str(age_str):
                    months_idx = parts.index('and') + 1 if 'and' in parts else 2
                    if months_idx < len(parts):
                        months = float(parts[months_idx])
                credit_history.append(years + (months / 12))
            except:
                continue
        
        averages['Credit_History_Age_Years'] = sum(credit_history) / len(credit_history) if credit_history else 5.0
    
    # Payment behavior
    if 'Payment_of_Min_Amount' in customer_data.columns:
        yes_count = (customer_data['Payment_of_Min_Amount'] == 'Yes').sum()
        total = len(customer_data)
        averages['Payment_of_Min_Amount'] = 1 if yes_count / total > 0.5 else 0
    
    # Occupation
    if 'Occupation' in customer_data.columns:
        mode_result = customer_data['Occupation'].mode()
        averages['Occupation'] = mode_result[0] if len(mode_result) > 0 else 'Unknown'
    
    # Calculate additional metrics
    original_debt = averages.get('Outstanding_Debt', 0)
    annual_income = averages.get('Annual_Income', 1)
    
    # Add loan amount to debt
    averages['Outstanding_Debt'] = original_debt + loan_amount
    
    # Calculate debt-to-income ratio
    debt_income_ratio = ((original_debt + loan_amount) / annual_income * 100) if annual_income > 0 else 0
    
    return averages, original_debt, debt_income_ratio

def predict_enhanced_risk(data_point, model, model_name):
    """Enhanced risk prediction with confidence"""
    if model is None:
        return {
            'risk_level': 'Medium Risk',
            'risk_percentage': '50.00%',
            'risk_description': f'{model_name} model unavailable',
            'approval_chance': '50-75%'
        }, {'level': 'Low', 'class': 'low-confidence'}
    
    try:
        # Prepare features
        numeric_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                           'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                           'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                           'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Years', 
                           'Payment_of_Min_Amount']
        categorical_cols = ['Occupation']
        
        # For B-Bank, add Source_Bank
        if model_name == 'B-BANK':
            categorical_cols.append('Source_Bank')
        
        complete_data_point = {}
        for key, value in data_point.items():
            complete_data_point[key] = value
        
        # Add defaults
        for feature in numeric_features:
            if feature not in complete_data_point:
                complete_data_point[feature] = 0
        
        for feature in categorical_cols:
            if feature not in complete_data_point:
                if feature == 'Source_Bank':
                    complete_data_point[feature] = 'COMBINED'
                else:
                    complete_data_point[feature] = 'Unknown'
        
        input_df = pd.DataFrame([complete_data_point])
        expected_features = numeric_features + categorical_cols
        input_df = input_df[expected_features]
        
        # Get prediction
        risk_level = model.predict(input_df)[0]
        risk_proba = model.predict_proba(input_df)[0]
        
        # Calculate confidence
        max_prob = np.max(risk_proba)
        confidence_level = 'High' if max_prob > 0.75 else 'Medium' if max_prob > 0.55 else 'Low'
        confidence_class = f'{confidence_level.lower()}-confidence'
        
        risk_percentage = risk_proba[2] * 100 if len(risk_proba) > 2 else risk_proba[-1] * 100
        
        return {
            'risk_level': risk_levels[risk_level]['name'],
            'risk_percentage': f"{risk_percentage:.2f}%",
            'risk_description': risk_levels[risk_level]['description'],
            'approval_chance': risk_levels[risk_level]['approval_chance']
        }, {'level': confidence_level, 'class': confidence_class}
        
    except Exception as e:
        print(f"Error in {model_name} prediction: {e}")
        return {
            'risk_level': 'Medium Risk',
            'risk_percentage': '50.00%',
            'risk_description': f'Error in {model_name} prediction',
            'approval_chance': '50-75%'
        }, {'level': 'Low', 'class': 'low-confidence'}

def generate_detailed_analysis(averages, bbank_risk, all_risks):
    """Generate detailed risk analysis with explanations"""
    analysis = {
        'positive_factors': [],
        'risk_concerns': [],
        'critical_factors': [],
        'financial_indicators': []
    }
    
    # Analyze financial indicators
    annual_income = averages.get('Annual_Income', 0)
    outstanding_debt = averages.get('Outstanding_Debt', 0)
    age = averages.get('Age', 0)
    credit_history = averages.get('Credit_History_Age_Years', 0)
    payment_behavior = averages.get('Payment_of_Min_Amount', 0)
    credit_utilization = averages.get('Credit_Utilization_Ratio', 0)
    delayed_payments = averages.get('Num_of_Delayed_Payment', 0)
    
    # Positive factors
    if annual_income > 50000:
        analysis['positive_factors'].append(f"Strong annual income of ${annual_income:,.2f} indicates good earning capacity")
    
    if age >= 25 and age <= 55:
        analysis['positive_factors'].append(f"Optimal age of {age} years shows financial stability period")
    
    if credit_history > 5:
        analysis['positive_factors'].append(f"Excellent credit history of {credit_history:.1f} years demonstrates long-term financial responsibility")
    
    if payment_behavior == 1:
        analysis['positive_factors'].append("Consistent minimum payment history shows reliable payment behavior")
    
    if credit_utilization < 30:
        analysis['positive_factors'].append(f"Low credit utilization of {credit_utilization:.1f}% indicates responsible credit management")
    
    # Risk concerns
    if delayed_payments > 3:
        analysis['risk_concerns'].append(f"Concerning payment delays: {delayed_payments} instances of delayed payments")
    
    if credit_utilization > 70:
        analysis['risk_concerns'].append(f"High credit utilization of {credit_utilization:.1f}% suggests potential financial stress")
    
    if annual_income < 30000:
        analysis['risk_concerns'].append(f"Lower income of ${annual_income:,.2f} may limit repayment capacity")
    
    # Critical factors
    debt_income_ratio = (outstanding_debt / annual_income * 100) if annual_income > 0 else 0
    if debt_income_ratio > 80:
        analysis['critical_factors'].append(f"CRITICAL: Debt-to-income ratio of {debt_income_ratio:.1f}% exceeds safe lending limits")
    
    if delayed_payments > 10:
        analysis['critical_factors'].append(f"CRITICAL: Excessive payment delays ({delayed_payments}) indicate severe payment issues")
    
    if credit_history < 1:
        analysis['critical_factors'].append("CRITICAL: Insufficient credit history for reliable risk assessment")
    
    # Financial indicators
    monthly_salary = averages.get('Monthly_Inhand_Salary', 0)
    if monthly_salary > 0:
        analysis['financial_indicators'].append(f"Monthly in-hand salary: ${monthly_salary:,.2f}")
    
    num_accounts = averages.get('Num_Bank_Accounts', 0)
    if num_accounts > 0:
        analysis['financial_indicators'].append(f"Banking relationship: {num_accounts} active bank accounts")
    
    investment = averages.get('Amount_invested_monthly', 0)
    if investment > 0:
        analysis['financial_indicators'].append(f"Investment activity: ${investment:,.2f} monthly investments show financial planning")
    
    # Ensure minimum content
    if not analysis['positive_factors']:
        analysis['positive_factors'].append("Customer has basic financial profile meeting minimum requirements")
    
    if not analysis['risk_concerns']:
        analysis['risk_concerns'].append("No significant risk concerns identified in current analysis")
    
    if not analysis['critical_factors']:
        analysis['critical_factors'].append("No critical risk factors detected")
    
    return analysis

def generate_recommendation_html(customer_data, averages, bbank_risk_pct, loan_amount, debt_income_ratio, loan_to_income_ratio):
    """Generate structured HTML for the recommendation"""
    
    annual_income = float(customer_data['Annual_Income'].iloc[0]) if 'Annual_Income' in customer_data.columns and pd.notna(customer_data['Annual_Income'].iloc[0]) else 0
    monthly_salary = averages.get('Monthly_Inhand_Salary', 0)
    credit_utilization = averages.get('Credit_Utilization_Ratio', 0)
    delayed_payments = averages.get('Num_of_Delayed_Payment', 0)
    credit_history_years = averages.get('Credit_History_Age_Years', 0)
    occupation = customer_data['Occupation'].iloc[0] if 'Occupation' in customer_data.columns else 'Unknown'
    age = averages.get('Age', 0)
    customer_name = customer_data['Name'].iloc[0] if 'Name' in customer_data.columns else 'N/A'

    if bbank_risk_pct < 15:
        # Premium Approval
        html = f'''
        <div class="recommendation-section">
            <div class="section-title">🏆 Executive Decision</div>
            <div class="action-badge approval-badge">IMMEDIATE APPROVAL - PREMIUM TIER</div>
        </div>
        
        <div class="recommendation-section">
            <div class="section-title">📊 Risk Assessment Summary</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Risk Score</div>
                    <div class="info-value">{bbank_risk_pct:.1f}% (Excellent)</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Credit Rating</div>
                    <div class="info-value">Premium Tier</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Default Probability</div>
                    <div class="info-value">Minimal Risk</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Processing Priority</div>
                    <div class="info-value">Fast Track</div>
                </div>
            </div>
        </div>

        <div class="recommendation-section">
            <div class="section-title">👤 Customer Profile Analysis</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Customer</div>
                    <div class="info-value">{customer_name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Annual Income</div>
                    <div class="info-value">${annual_income:,.2f}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Occupation</div>
                    <div class="info-value">{occupation}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Credit History</div>
                    <div class="info-value">{credit_history_years:.1f} years</div>
                </div>
            </div>
        </div>

        <div class="recommendation-section">
            <div class="section-title">✅ Key Strengths</div>
            <ul class="strengths-list">
                <li class="strength-item">Outstanding income capacity with ${annual_income:,.2f} annual earnings</li>
                <li class="strength-item">Excellent credit utilization at {credit_utilization:.1f}%</li>
                <li class="strength-item">Strong payment history with only {delayed_payments} delays</li>
                <li class="strength-item">Optimal age demographic at {age} years</li>
                <li class="strength-item">Conservative loan request at {loan_to_income_ratio:.1f}% of income</li>
            </ul>
        </div>

        <div class="recommendation-section">
            <div class="section-title">💰 Loan Package Details</div>
            <div class="terms-package">
                <div class="package-title">🏅 PREMIUM TERMS PACKAGE</div>
                <div class="terms-grid">
                    <div class="term-item">
                        <div class="term-label">Interest Rate</div>
                        <div class="term-value">Prime - 0.5%<br>(Est. 3.5% APR)</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Loan Amount</div>
                        <div class="term-value">${loan_amount:,.2f}<br>(Full Amount)</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Repayment Term</div>
                        <div class="term-value">5-7 years<br>(Customer Choice)</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Collateral</div>
                        <div class="term-value">None Required</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Processing Time</div>
                        <div class="term-value">24 hours<br>Expedited</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Benefits</div>
                        <div class="term-value">Rate Protection<br>Early Payoff Option</div>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    elif bbank_risk_pct < 25:
        # Standard Approval
        html = f'''
        <div class="recommendation-section">
            <div class="section-title">✅ Executive Decision</div>
            <div class="action-badge approval-badge">STANDARD APPROVAL</div>
        </div>
        
        <div class="recommendation-section">
            <div class="section-title">📊 Risk Assessment Summary</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Risk Score</div>
                    <div class="info-value">{bbank_risk_pct:.1f}% (Low Risk)</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Credit Rating</div>
                    <div class="info-value">Standard Approval</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Default Probability</div>
                    <div class="info-value">Low Risk</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Monitoring Level</div>
                    <div class="info-value">Standard</div>
                </div>
            </div>
        </div>

        <div class="recommendation-section">
            <div class="section-title">💰 Financial Metrics Analysis</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Loan-to-Income</div>
                    <div class="info-value">{loan_to_income_ratio:.1f}% ({'Conservative' if loan_to_income_ratio < 20 else 'Reasonable' if loan_to_income_ratio < 35 else 'Moderate'})</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Credit Utilization</div>
                    <div class="info-value">{credit_utilization:.1f}% ({'Excellent' if credit_utilization < 30 else 'Acceptable'})</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Payment Delays</div>
                    <div class="info-value">{delayed_payments} instances</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Debt-to-Income</div>
                    <div class="info-value">{debt_income_ratio:.1f}%</div>
                </div>
            </div>
        </div>

        <div class="recommendation-section">
            <div class="section-title">✅ Positive Indicators</div>
            <ul class="strengths-list">
                <li class="strength-item">Adequate repayment capacity with ${annual_income:,.2f} income</li>
                <li class="strength-item">{'Excellent' if credit_history_years > 5 else 'Adequate'} credit track record ({credit_history_years:.1f} years)</li>
                <li class="strength-item">{'Strong' if debt_income_ratio < 40 else 'Manageable'} debt management profile</li>
                <li class="strength-item">Stable financial profile and employment</li>
            </ul>
        </div>

        <div class="recommendation-section">
            <div class="section-title">💰 Loan Package Details</div>
            <div class="terms-package">
                <div class="package-title">📋 STANDARD TERMS PACKAGE</div>
                <div class="terms-grid">
                    <div class="term-item">
                        <div class="term-label">Interest Rate</div>
                        <div class="term-value">Standard Rate<br>(Est. 4.5% APR)</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Loan Amount</div>
                        <div class="term-value">${loan_amount:,.2f}<br>(Full Amount)</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Repayment Term</div>
                        <div class="term-value">5 years<br>Standard</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Collateral</div>
                        <div class="term-value">None Required</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Processing Time</div>
                        <div class="term-value">3-5 business days</div>
                    </div>
                    <div class="term-item">
                        <div class="term-label">Monitoring</div>
                        <div class="term-value">Standard Protocols</div>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    else:
        # Higher risk scenarios (conditional approval, review, or decline)
        html = f'''
        <div class="recommendation-section">
            <div class="section-title">⚠️ Executive Decision</div>
            <div class="action-badge conditional-badge">REQUIRES FURTHER REVIEW</div>
        </div>
        
        <div class="recommendation-section">
            <div class="section-title">📊 Risk Assessment Summary</div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Risk Score</div>
                    <div class="info-value">{bbank_risk_pct:.1f}% (Higher Risk)</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Status</div>
                    <div class="info-value">Manual Review Required</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Debt-to-Income</div>
                    <div class="info-value">{debt_income_ratio:.1f}%</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Recommendation</div>
                    <div class="info-value">Enhanced Terms</div>
                </div>
            </div>
        </div>
        '''
    
    return html

@app.route('/process', methods=['POST'])
def process():
    try:
        customer_id = request.form.get('customer_id')
        loan_amount = float(request.form.get('loan_amount'))
        
        # Find customer
        customer_data, data_source = get_customer_data(customer_id)
        
        if customer_data is None:
            return jsonify({
                'success': False,
                'message': f'Customer ID "{customer_id}" not found in any banking system records.'
            })
        
        # Calculate enhanced averages
        averages, original_debt, debt_income_ratio = calculate_enhanced_averages(customer_data, loan_amount)
        
        # Get predictions from all models
        risks = {}
        confidences = {}
        
        for name in ['sb', 'pb', 'fnb', 'bbank']:
            if models.get(name):
                risks[name], confidences[name] = predict_enhanced_risk(averages, models[name], name.upper())
            else:
                risks[name] = {
                    'risk_level': 'Medium Risk',
                    'risk_percentage': '50.00%',
                    'risk_description': f'{name.upper()} model unavailable',
                    'approval_chance': '50-75%'
                }
                confidences[name] = {'level': 'Low', 'class': 'low-confidence'}
        
        # Generate detailed analysis
        detailed_analysis = generate_detailed_analysis(averages, risks['bbank'], risks)
        
        # Model agreement analysis
        risk_percentages = [float(risks[name]['risk_percentage'].replace('%', '')) for name in ['sb', 'pb', 'fnb']]
        bbank_percentage = float(risks['bbank']['risk_percentage'].replace('%', ''))
        
        avg_other_models = sum(risk_percentages) / len(risk_percentages) if risk_percentages else 50
        difference = abs(bbank_percentage - avg_other_models)
        
        if difference < 10:
            model_agreement = f"High agreement: B-Bank decision aligns closely with other models (±{difference:.1f}%)"
        elif difference < 25:
            model_agreement = f"Moderate agreement: B-Bank shows some variation from other models (±{difference:.1f}%)"
        else:
            model_agreement = f"Significant variation: B-Bank assessment differs substantially from other models (±{difference:.1f}%)"
        
        # Calculate loan-to-income ratio
        annual_income = float(customer_data['Annual_Income'].iloc[0]) if 'Annual_Income' in customer_data.columns and pd.notna(customer_data['Annual_Income'].iloc[0]) else 0
        loan_to_income_ratio = (loan_amount / annual_income * 100) if annual_income > 0 else 0
        
        # Generate structured HTML recommendation
        bbank_risk_pct = float(risks['bbank']['risk_percentage'].replace('%', ''))
        final_recommendation_html = generate_recommendation_html(
            customer_data, averages, bbank_risk_pct, loan_amount, 
            debt_income_ratio, loan_to_income_ratio
        )
        
        # Customer details
        customer_details = {
            'name': customer_data['Name'].iloc[0] if 'Name' in customer_data.columns else 'N/A',
            'age': str(int(float(customer_data['Age'].iloc[0]))) if 'Age' in customer_data.columns and pd.notna(customer_data['Age'].iloc[0]) else 'N/A',
            'occupation': customer_data['Occupation'].iloc[0] if 'Occupation' in customer_data.columns else 'N/A',
            'annual_income': f"${float(customer_data['Annual_Income'].iloc[0]):,.2f}" if 'Annual_Income' in customer_data.columns and pd.notna(customer_data['Annual_Income'].iloc[0]) else 'N/A',
            'original_debt': f"${original_debt:,.2f}",
            'requested_loan': f"${loan_amount:,.2f}",
            'total_debt': f"${(original_debt + loan_amount):,.2f}",
            'debt_income_ratio': f"{debt_income_ratio:.1f}%"
        }
        
        return jsonify({
            'success': True,
            'customer': customer_details,
            'sb_risk': risks['sb'],
            'pb_risk': risks['pb'],
            'fnb_risk': risks['fnb'],
            'bbank_risk': risks['bbank'],
            'bbank_confidence': confidences['bbank'],
            'model_agreement': model_agreement,
            'detailed_analysis': detailed_analysis,
            'final_recommendation_html': final_recommendation_html
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        })

if __name__ == '__main__':
    import os
    print("\n🏦 Advanced Multi-Model Risk Assessment System")
    print("=" * 50)
    model_display_names = {'sb': 'SB', 'pb': 'PB', 'fnb': 'FNB', 'bbank': 'B-Bank'}
    for key, display_name in model_display_names.items():
        status = "✓ Ready" if models.get(key) else "✗ Failed"
        if model_stats.get(key) and isinstance(model_stats[key], dict) and 'accuracy' in model_stats[key]:
            accuracy = f"({model_stats[key]['accuracy']:.3f})" if isinstance(model_stats[key]['accuracy'], float) else f"({model_stats[key]['accuracy']})"
        else:
            accuracy = ""
        print(f"{display_name} Model: {status} {accuracy}")
    
    print("\n💾 Model files will be saved for faster loading next time")

    # 👇 Updated to work on Render
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
