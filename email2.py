import streamlit as st
import joblib
import numpy as np
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize history if not present
if "history" not in st.session_state:
    st.session_state.history = []

# Load models with error handling
try:
    model1 = joblib.load("naive_bayes_model.joblib")
    model2 = joblib.load("logistic_regression_model.joblib")
    model3 = joblib.load("svm.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def spam_checking(new_email):
    return vectorizer.transform([new_email])

def naive_bayes(new_email):
    x_new = spam_checking(new_email)
    prediction = model1.predict(x_new)[0]
    confidence = model1.predict_proba(x_new)[0][1]
    return prediction, confidence

def logistic_regression(new_email):
    x_new = spam_checking(new_email)
    confidence = model2.predict_proba(x_new)[0][1]
    prediction = 1 if confidence >= 0.56 else 0
    return prediction, confidence

def svm(new_email):
    x_new = spam_checking(new_email)
    prediction = model3.predict(x_new)[0]
    decision_score = model3.decision_function(x_new)[0]
    confidence = 1 / (1 + np.exp(-decision_score))  # Sigmoid
    return prediction, confidence

# UI Header
st.title("📧 Spam Email Detector")
st.markdown("Check if an email is **Spam** or **Not Spam** using 3 different ML models!")

# User Information Section
st.markdown("### 👤 Your Information")
user_email = st.text_input(
    "Your Email Address:",
    placeholder="example@email.com",
    help="We'll use this to track your spam check history"
)

# Email Input with better instructions
st.markdown("### ✉️ Paste Email Content Below")
st.caption("Copy and paste the email text you received (subject + body) into the box below:")
new_email = st.text_area(
    "Email content:",
    placeholder="Example:\n\nSubject: Congratulations! You've won $1,000,000!\n\nDear Winner, click here to claim your prize...",
    height=200,
    label_visibility="collapsed"
)

# Button to check spam
if st.button("🔍 Check Spam", type="primary"):
    if user_email.strip() == "":
        st.warning("⚠️ Please enter your email address first.")
    elif new_email.strip() == "":
        st.warning("⚠️ Please enter some email text to check.")
    else:
        with st.spinner("Analyzing email..."):
            nb_pred, nb_conf = naive_bayes(new_email)
            lr_pred, lr_conf = logistic_regression(new_email)
            svm_pred, svm_conf = svm(new_email)
            
            spam_votes = nb_pred + lr_pred + svm_pred
            avg_conf = (nb_conf + lr_conf + svm_conf) / 3
        
        # Show each model's result
        st.subheader("🔍 Results from Each Model:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🧠 Naive Bayes",
                "Spam" if nb_pred else "Not Spam",
                f"{round(nb_conf*100, 2)}%"
            )
        
        with col2:
            st.metric(
                "📈 Logistic Regression",
                "Spam" if lr_pred else "Not Spam",
                f"{round(lr_conf*100, 2)}%"
            )
        
        with col3:
            st.metric(
                "📊 SVM",
                "Spam" if svm_pred else "Not Spam",
                f"{round(svm_conf*100, 2)}%"
            )
        
        # Final decision
        st.markdown("---")
        if spam_votes >= 2:
            final_result = "SPAM"
            st.error(f"🚨 **Final Decision: {final_result}**")
            st.caption(f"Average Confidence: {round(avg_conf*100, 2)}%")
        else:
            final_result = "NOT SPAM"
            st.success(f"✅ **Final Decision: {final_result}**")
            st.caption(f"Average Confidence: {round((1 - avg_conf)*100, 2)}%")
        
        # Save to session history
        st.session_state.history.append({
            "user_email": user_email,
            "email": new_email[:100] + "..." if len(new_email) > 100 else new_email,
            "result": final_result,
            "confidence": round(avg_conf * 100, 2)
        })

# History section
st.markdown("---")
if st.button("📜 View History"):
    if len(st.session_state.history) == 0:
        st.info("No history yet. Try classifying some emails!")
    else:
        st.subheader("📜 Previous Emails Checked:")
        for idx, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"#{idx} - {item['user_email']} - {item['result']} ({item['confidence']}%)"):
                st.text(item['email'])

# Clear history button
if len(st.session_state.history) > 0:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("👨‍💻 Created by **Rajan**")