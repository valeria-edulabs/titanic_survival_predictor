# app.py
"""
Streamlit web application for the Titanic Survival Prediction project.
Allows users to train, evaluate, and run predictions with a safe-deployment workflow.
"""
import streamlit as st
import pandas as pd
import os

from model_pipeline import (
    train_model,
    evaluate_model,
    run_prediction,
    promote_model_to_production,
    validate_training_data  # Import the new function
)
import config

# --- App Layout ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Predictor")
st.write("An app to demonstrate a production-ready ML workflow with model validation.")

st.sidebar.title("Actions")
action = st.sidebar.radio("Choose an action:", ("Predict Survival", "Train New Model", "Evaluate Production Model"))

# ======================================================================================
#                            ACTION: PREDICT SURVIVAL
# ======================================================================================
if action == "Predict Survival":
    st.header("Check a Passenger's Survival Odds")
    st.markdown("Enter passenger details to predict survival using the **production model**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare ($)", min_value=0.0, max_value=1000.0, value=50.0)
    with col3:
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    if st.button("Predict 🔮", use_container_width=True):
        try:
            input_data = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch,
                                        'Fare': fare, 'Embarked': embarked}])
            prediction, probability = run_prediction(input_data)

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"**Outcome: Likely Survived** (Probability: {probability:.2%})")
                st.balloons()
            else:
                st.error(f"**Outcome: Likely Did Not Survive** (Probability of survival: {probability:.2%})")
        except FileNotFoundError:
            st.error(f"Production model not found. Please train a model first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# ======================================================================================
#                            ACTION: TRAIN NEW MODEL
# ======================================================================================
elif action == "Train New Model":
    st.header("Train a New Candidate Model")

    # --- 1. Data Source Selection ---
    st.subheader("1. Select Training Data")
    data_source = st.radio("Choose data source:", ("Use default data", "Upload new CSV"), horizontal=True)

    training_data_path = config.TRAIN_DATA_PATH
    can_train = True  # Flag to control button state

    # --- MODIFIED SECTION FOR VALIDATION ---
    if data_source == "Upload new CSV":
        uploaded_file = st.file_uploader("Upload your training CSV", type="csv")
        if uploaded_file is not None:
            try:
                # Read the uploaded file into a DataFrame
                df_uploaded = pd.read_csv(uploaded_file)

                # Validate the DataFrame
                is_valid, errors = validate_training_data(df_uploaded)

                if is_valid:
                    st.success("✅ CSV structure is valid and ready for training.")
                    # Save the valid DataFrame to a temporary file to be used by the training function
                    temp_upload_path = "data/temp_upload.csv"
                    os.makedirs("data", exist_ok=True)
                    df_uploaded.to_csv(temp_upload_path, index=False)
                    training_data_path = temp_upload_path
                else:
                    st.error("❌ Invalid CSV file. Please fix the following issues:")
                    with st.expander("See validation errors"):
                        for error in errors:
                            st.write(f"- {error}")
                    can_train = False  # Disable training if validation fails
            except Exception as e:
                st.error(f"An error occurred while reading or validating the file: {e}")
                can_train = False
        else:
            can_train = False  # Disable training if no file is uploaded

    # --- 2. Model Configuration ---
    st.subheader("2. Configure Model Parameters")
    with st.expander("Hyperparameter Settings"):
        C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01, help="Lower C = stronger regularization.")
        model_params = {'C': C}

    # --- 3. Training and Comparison ---
    st.subheader("3. Train and Evaluate")
    # The button is now disabled if can_train is False
    if st.button("Train New Candidate Model 🏋️", use_container_width=True, disabled=not can_train):
        with st.spinner("Training and evaluating candidate model..."):
            try:
                candidate_model_path = train_model(training_data_path, model_params)
                st.session_state['candidate_metrics'] = evaluate_model(candidate_model_path)

                if os.path.exists(config.MODEL_PATH):
                    st.session_state['production_metrics'] = evaluate_model(config.MODEL_PATH)
                else:
                    st.session_state['production_metrics'] = None

                st.session_state['show_comparison'] = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state['show_comparison'] = False

    # --- 4. Display Comparison and Promote ---
    if st.session_state.get('show_comparison', False):
        st.subheader("4. Model Comparison")
        # (This section remains unchanged)
        prod_metrics = st.session_state.get('production_metrics')
        cand_metrics = st.session_state.get('candidate_metrics')
        if prod_metrics:
            col1_header, col2_header = st.columns(2)
            with col1_header:
                st.markdown("#### Production Model")
                st.metric("Accuracy", f"{prod_metrics['accuracy']:.4f}")
            with col2_header:
                st.markdown("#### Candidate Model")
                st.metric("Accuracy", f"{cand_metrics['accuracy']:.4f}",
                          delta=f"{cand_metrics['accuracy'] - prod_metrics['accuracy']:.4f}")
            st.markdown("---")
            st.markdown("###### Classification Reports")
            col1_report, col2_report = st.columns(2)
            with col1_report:
                prod_report_df = pd.DataFrame(prod_metrics['classification_report']).transpose()
                st.dataframe(prod_report_df)
            with col2_report:
                cand_report_df = pd.DataFrame(cand_metrics['classification_report']).transpose()
                st.dataframe(cand_report_df)
            if cand_metrics['accuracy'] < prod_metrics['accuracy']:
                st.warning("The candidate model performs worse than the current production model.", icon="⚠️")
        else:
            st.info(
                "No existing production model. The candidate model will become the first production model upon promotion.")
            st.metric("Candidate Model Accuracy", f"{cand_metrics['accuracy']:.4f}")
            st.markdown("###### Classification Report")
            cand_report_df = pd.DataFrame(cand_metrics['classification_report']).transpose()
            st.dataframe(cand_report_df)
        if st.button("Promote Candidate to Production ✅", use_container_width=True):
            try:
                message = promote_model_to_production()
                st.success(message)
                st.session_state['show_comparison'] = False
            except Exception as e:
                st.error(f"Failed to promote model: {e}")

# ======================================================================================
#                        ACTION: EVALUATE PRODUCTION MODEL
# ======================================================================================
elif action == "Evaluate Production Model":
    st.header("Evaluate the Current Production Model")
    st.markdown(f"This evaluates the model at `{config.MODEL_PATH}` against the default test set.")

    if st.button("Run Evaluation 📊", use_container_width=True):
        with st.spinner("Evaluating model..."):
            try:
                metrics = evaluate_model(config.MODEL_PATH)
                st.subheader("Production Model Metrics")
                st.metric(label="**Accuracy**", value=f"{metrics['accuracy']:.4f}")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df)
            except FileNotFoundError:
                st.error("No production model found. Please train a model first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")