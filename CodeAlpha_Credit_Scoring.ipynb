import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. CREATE MOCK DATA (In a real project, replace this with pd.read_csv('your_file.csv'))
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 28],
    'income': [50000, 80000, 60000, 120000, 20000, 90000, 110000, 75000, 45000, 30000],
    'loan_amount': [10000, 5000, 15000, 20000, 5000, 2000, 30000, 10000, 8000, 12000],
    'credit_score': [600, 750, 680, 800, 550, 720, 780, 710, 640, 590],
    'is_default': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1] # 1 = Default (Bad), 0 = Paid (Good)
}
df = pd.DataFrame(data)

# 2. FEATURE ENGINEERING & PREPROCESSING
# Defining our features (X) and what we want to predict (y)
feature_names = ['age', 'income', 'loan_amount', 'credit_score']
X = df[feature_names]
y = df['is_default']

# Scaling data (Crucial for Logistic Regression to perform well)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. MODEL TRAINING
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test)
print("--- MODEL ASSESSMENT ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 6. REAL-TIME PREDICTION SYSTEM
print("\n--- TEST NEW APPLICANT ---")
def predict_credit():
    try:
        age = float(input("Enter Age: "))
        income = float(input("Enter Annual Income: "))
        loan = float(input("Enter Loan Amount: "))
        score = float(input("Enter Credit Score: "))

        # Put input into a DataFrame to keep feature names consistent
        new_data = pd.DataFrame([[age, income, loan, score]], columns=feature_names)
        
        # Must scale the input exactly like we scaled the training data
        new_data_scaled = scaler.transform(new_data)
        
        prediction = model.predict(new_data_scaled)
        probability = model.predict_proba(new_data_scaled)[0][0] # Probability of being '0' (Good)

        if prediction[0] == 0:
            print(f"\nResult: LOAN APPROVED (Confidence: {probability:.2%})")
        else:
            print(f"\nResult: LOAN DENIED (Risk too high)")
    except Exception as e:
        print(f"Error: {e}")

predict_credit()
