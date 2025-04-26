# credit_card_fraud_detection_app.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv(r'c:\credit_card_fraud_detection_app\creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data_balanced = pd.concat([legit_sample, fraud], axis=0)

# Features and labels
X = data_balanced.drop(columns='Class', axis=1)
y = data_balanced['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model with more iterations
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model accuracy
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit App
st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details to check if it's **legitimate** or **fraudulent**.")
st.write(f"🧠Model Accuracy on Training Set: {train_acc:.2f}")
st.write(f"🧪 Model Accuracy on Test Set: {test_acc:.2f}")

# Feature input
st.markdown("**Enter all input features (space-separated):**")
input_str = st.text_input(f"Enter {X.shape[1]} numerical features (V1 to V28, Time, Amount):")

# Predict button
if st.button("Submit"):
    input_list = input_str.strip().split()
    
    if len(input_list) != X.shape[1]:
        st.error(f"❌ Please enter exactly {X.shape[1]} values.")
    else:
        try:
            input_features = np.array(input_list, dtype=np.float64).reshape(1, -1)
            prediction = model.predict(input_features)
            if prediction[0] == 0:
                st.success("✅ The transaction is **Legitimate**.")
            else:
                st.error("⚠️ The transaction is **Fraudulent**.")
        except ValueError:
            st.error("❌ Invalid input! Make sure all values are numeric.")
