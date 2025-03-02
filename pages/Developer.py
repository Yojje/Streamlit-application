import streamlit as st
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import os

# Authentication credentials
VALID_USERNAME = "yoje"
VALID_PASSWORD = "yoje123"

# File path for storing feedback
FEEDBACK_FILE = "data/feedback.csv"

def ensure_feedback_file():
    """Create feedback file if it doesn't exist"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Feedback'])

def save_feedback(feedback):
    """Save new feedback to CSV file"""
    ensure_feedback_file()
    with open(FEEDBACK_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), feedback])

def load_feedback():
    """Load all feedback from CSV file"""
    ensure_feedback_file()
    try:
        df = pd.read_csv(FEEDBACK_FILE)
        if df.empty:
            return pd.DataFrame(columns=['Timestamp', 'Feedback'])
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=['Timestamp', 'Feedback'])

def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        st.title("Developer Login Required")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    return True

def show_developer_page():
    if not check_auth():
        return

    st.title("Developer Tools")
    
    # Save new feedback if available in session state
    if 'feedback' in st.session_state and st.session_state.feedback:
        save_feedback(st.session_state.feedback)
        # Clear the feedback from session state after saving
        st.session_state.feedback = ""

    # Load and display all feedback
    df = load_feedback()
    if not df.empty:
        st.subheader("All User Feedback")
        st.dataframe(df, use_container_width=True)
        
        # Export option
        if st.button("Export Feedback to CSV"):
            df.to_csv("feedback_export.csv", index=False)
            st.success("Feedback exported to feedback_export.csv")
    else:
        st.info("No feedback submitted yet")

if __name__ == "__main__":
    show_developer_page()