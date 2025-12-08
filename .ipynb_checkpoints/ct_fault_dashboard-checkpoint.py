# ct_fault_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =========================
#  Load & prepare data
# =========================
@st.cache_data
def load_data(path="ct_fault_dataset.csv"):
    df = pd.read_csv(path)
    return df


@st.cache_data
def split_data(df, test_size=0.2, random_state=42):
    feature_cols = [c for c in df.columns if c != "fault_type"]
    X = df[feature_cols]
    y = df["fault_type"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, feature_cols, le


# =========================
#  Train model
# =========================
def get_model(model_name, params):
    if model_name == "Logistic Regression":
        model = LogisticRegression(
            C=params["C"],
            max_iter=1000,
            n_jobs=-1,
            multi_class="auto"
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "SVM (RBF)":
        model = SVC(
            C=params["C"],
            gamma=params["gamma"],
            probability=True
        )
    elif model_name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"]
        )
    else:
        model = RandomForestClassifier(random_state=42)
    return model


def train_model(model_name, params, X_train, X_test, y_train, y_test, scale_features):
    # Optional scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model = get_model(model_name, params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, acc, cm, y_pred


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# =========================
#  Streamlit UI
# =========================
def main():
    st.set_page_config(
        page_title="CT Fault Detection - AML Dashboard",
        layout="wide"
    )

    st.title("⚡ CT Fault Detection – AML Dashboard")
    st.write("Interactive dashboard for your **ct_fault_dataset** transformer fault model.")

    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test, feature_cols, label_encoder = split_data(df)

    # Sidebar – model selection
    st.sidebar.header("Model Settings")

    model_name = st.sidebar.selectbox(
        "Choose model",
        ["Random Forest", "Logistic Regression", "SVM (RBF)", "KNN"]
    )

    scale_features = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)

    params = {}
    if model_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
        max_depth = st.sidebar.slider("max_depth", 2, 20, 10, step=1)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif model_name == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization strength)", 0.01, 10.0, 1.0)
        params["C"] = C
    elif model_name == "SVM (RBF)":
        C = st.sidebar.slider("C", 0.1, 10.0, 1.0)
        gamma = st.sidebar.slider("gamma", 0.001, 1.0, 0.1)
        params["C"] = C
        params["gamma"] = gamma
    elif model_name == "KNN":
        n_neighbors = st.sidebar.slider("n_neighbors (K)", 1, 25, 5)
        params["n_neighbors"] = n_neighbors

    st.sidebar.markdown("---")
    train_button = st.sidebar.button("🚀 Train / Retrain Model")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Overview", "🤖 Model Training", "📈 Evaluation", "🔮 Prediction"]
    )

    # ------------- Tab 1: Data Overview -------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.markdown("**Shape:**")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.markdown("**Summary Statistics (features)**")
        st.dataframe(df[feature_cols].describe())

        st.markdown("**Fault Type Distribution**")
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.bar_chart(df["fault_type"].value_counts())

        with col2:
            fig, ax = plt.subplots()
            df["fault_type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Class Proportions")
            st.pyplot(fig)

    # Train model when button pressed (or once by default)
    if train_button or "model" not in st.session_state:
        model, scaler, acc, cm, y_pred = train_model(
            model_name, params, X_train, X_test, y_train, y_test, scale_features
        )
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.acc = acc
        st.session_state.cm = cm
        st.session_state.y_pred = y_pred
        st.session_state.model_name = model_name
        st.session_state.params = params

    # If no training yet, train once with default sidebar settings
    if "model" not in st.session_state:
        model, scaler, acc, cm, y_pred = train_model(
            model_name, params, X_train, X_test, y_train, y_test, scale_features
        )
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.acc = acc
        st.session_state.cm = cm
        st.session_state.y_pred = y_pred
        st.session_state.model_name = model_name
        st.session_state.params = params

    model = st.session_state.model
    scaler = st.session_state.scaler
    acc = st.session_state.acc
    cm = st.session_state.cm
    y_pred = st.session_state.y_pred

    class_names = list(label_encoder.classes_)

    # ------------- Tab 2: Model Training -------------
    with tab2:
        st.subheader("Training Configuration")

        st.write(f"**Selected model:** {st.session_state.model_name}")
        st.write("**Parameters:**")
        st.json(st.session_state.params)

        st.write("**Train/Test Split:** 80% train, 20% test")
        st.write("**Feature scaling:** ", "Enabled" if scale_features else "Disabled")

        st.success("Model is trained and ready for evaluation & prediction.")

    # ------------- Tab 3: Evaluation -------------
    with tab3:
        st.subheader("Model Evaluation")

        st.metric("Accuracy on Test Set", f"{acc*100:.2f}%")

        st.markdown("### Confusion Matrix")
        plot_confusion_matrix(cm, class_names)

        st.markdown("### Classification Report")
        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

    # ------------- Tab 4: Prediction -------------
    with tab4:
        st.subheader("Make Predictions")

        pred_mode = st.radio(
            "Prediction mode:",
            ["Single sample (manual input)", "Batch prediction (upload CSV)"]
        )

        # Single-sample prediction
        if pred_mode == "Single sample (manual input)":
            st.markdown("Enter feature values:")

            # Use data stats for sensible ranges
            desc = df[feature_cols].describe()

            user_input = {}
            cols = st.columns(3)
            for i, col in enumerate(feature_cols):
                c = cols[i % 3]
                col_min = float(desc.loc["min", col])
                col_max = float(desc.loc["max", col])
                col_mean = float(desc.loc["mean", col])

                user_input[col] = c.slider(
                    col,
                    min_value=col_min,
                    max_value=col_max,
                    value=col_mean,
                    step=(col_max - col_min) / 100.0
                )

            if st.button("Predict Fault Type"):
                sample = pd.DataFrame([user_input])

                if scaler is not None:
                    sample_scaled = scaler.transform(sample)
                else:
                    sample_scaled = sample

                pred_label_idx = model.predict(sample_scaled)[0]
                pred_label = label_encoder.inverse_transform([pred_label_idx])[0]

                st.success(f"Predicted fault type: **{pred_label}**")

        # Batch prediction
        else:
            st.markdown("Upload a CSV file with the same feature columns:")
            st.write(feature_cols)

            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)

                # Ensure only feature columns are used
                batch_X = batch_df[feature_cols]

                if scaler is not None:
                    batch_X_scaled = scaler.transform(batch_X)
                else:
                    batch_X_scaled = batch_X

                batch_pred_idx = model.predict(batch_X_scaled)
                batch_pred_labels = label_encoder.inverse_transform(batch_pred_idx)

                result_df = batch_df.copy()
                result_df["predicted_fault_type"] = batch_pred_labels

                st.write("Preview of predictions:")
                st.dataframe(result_df.head())

                # Download link
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=csv,
                    file_name="ct_fault_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
