import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for a clean app interface
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="FraudLens: Simple Screening",
    page_icon="🔎",
    layout="wide"
)

# --- Caching Functions ---
# @st.cache_data runs ONCE to load the data.
# @st.cache_resource runs ONCE to train the model.

@st.cache_data
def load_data(filepath):
    """Loads the loan dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please make sure it's in the same folder as app.py.")
        return None

@st.cache_resource
def train_model(df):
    """Preprocesses data and trains a RandomForest model."""
    
    # --- ERROR LINE REMOVED ---
    # The "st.toast(...)" line that was here has been removed.
    # We cannot call st elements inside a cached function.

    # 1. Preprocessing (as you did in your notebook)
    df_processed = pd.get_dummies(df, columns=['purpose'], drop_first=True)
    
    # 2. Define Features (X) and Target (y)
    X = df_processed.drop('not.fully.paid', axis=1)
    y = df_processed['not.fully.paid']
    
    # 3. Get column order and scaler
    feature_columns = X.columns
    
    # 3. Scale the data (Good practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    # 5. Return everything we need for prediction
    return model, scaler, feature_columns, df['purpose'].unique()

# --- Main Application ---

# Load the data
loan_df = load_data('loan_data.csv')

# Only run the app if the data is successfully loaded
if loan_df is not None:
    
    # This is the *correct* way to show a loading message:
    # We show it *before* calling the cached function.
    with st.spinner("Training model on first startup... This may take a moment."):
        model, scaler, feature_columns, purpose_categories = train_model(loan_df)

    # --- UI Layout ---
    st.title("🔎 FraudLens: Data-Driven Loan Application Screening")
    st.markdown("""
    This app predicts the **risk of a loan not being fully repaid**. 
    Enter the applicant's details in the sidebar to get a real-time risk assessment.
    """)

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Applicant Information")

    inputs = {} 
    inputs['credit.policy'] = st.sidebar.selectbox("Credit Policy Status", [1, 0], format_func=lambda x: "1: Meets Policy" if x == 1 else "0: Does Not Meet")
    inputs['fico'] = st.sidebar.slider("FICO Score", 600, 850, 710)
    inputs['int.rate'] = st.sidebar.slider("Interest Rate (e.g., 0.12 for 12%)", 0.05, 0.25, 0.12)
    inputs['installment'] = st.sidebar.number_input("Monthly Installment ($)", min_value=15.0, value=300.0, step=10.0)
    inputs['log.annual.inc'] = st.sidebar.number_input("Log of Annual Income", min_value=7.0, max_value=15.0, value=11.0, step=0.1)
    inputs['dti'] = st.sidebar.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    inputs['days.with.cr.line'] = st.sidebar.number_input("Days with Credit Line", min_value=100.0, value=4500.0, step=100.0)
    inputs['revol.bal'] = st.sidebar.number_input("Revolving Balance ($)", min_value=0, value=10000, step=100)
    inputs['revol.util'] = st.sidebar.slider("Revolving Line Utilization (%)", 0.0, 120.0, 50.0, step=0.1)
    inputs['inq.last.6mths'] = st.sidebar.slider("Inquiries in Last 6 Months", 0, 10, 1) 
    inputs['delinq.2yrs'] = st.sidebar.slider("Delinquencies in Last 2 Years", 0, 5, 0) 
    inputs['pub.rec'] = st.sidebar.slider("Public Records", 0, 5, 0) 
    
    selected_purpose = st.sidebar.selectbox("Purpose of Loan", purpose_categories)

    # --- Prediction Button and Logic ---
    if st.sidebar.button("Screen Applicant", type="primary"):
        
        input_df = pd.DataFrame([inputs])
        
        for p in purpose_categories:
            column_name = f"purpose_{p}"
            if column_name in feature_columns:
                input_df[column_name] = (selected_purpose == p)
            
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        input_scaled = scaler.transform(input_df)
        
        prediction_proba = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]

        # --- Display results ---
        st.subheader("Screening Result")
        
        if prediction == 0:
            st.success("**Prediction: LOW RISK (Likely to Repay)**")
            prob = prediction_proba[0]
            st.metric(label="Confidence (Low Risk)", value=f"{prob*100:.2f}%")
            st.progress(prob)
        else:
            st.error("**Prediction: HIGH RISK (Likely to Not Repay)**")
            prob = prediction_proba[1]
            st.metric(label="Confidence (High Risk)", value=f"{prob*100:.2f}%")
            st.progress(prob)

        with st.expander("Show Input Data Sent to Model (Advanced)"):
            st.dataframe(input_df)

