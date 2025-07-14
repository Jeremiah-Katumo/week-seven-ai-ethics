import streamlit as st
import pandas as pd
import numpy as np
import janitor
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="COMPAS Bias Audit", layout="wide")
st.title("📊 COMPAS Advanced Bias Audit Dashboard")

# -------------------- Upload Dataset --------------------
st.sidebar.header("🗂️ Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload compas_scores.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = janitor.clean_names(df)  # Clean column names
    st.write("📄 Sample Data", df.head())

    # -------------------- Column Selection --------------------
    st.sidebar.subheader("⚙️ Column Settings")
    label_col = st.sidebar.selectbox("Select Label Column", df.columns, index=df.columns.get_loc("two_year_recid") if "two_year_recid" in df.columns else 0)
    protected_attr = st.sidebar.selectbox("Select Protected Attribute", df.columns, index=df.columns.get_loc("race") if "race" in df.columns else 0)

    # Show class balance
    st.sidebar.info(f"Label Balance: {df[label_col].value_counts().to_dict()}")

    # -------------------- Feature Selection --------------------
    drop_cols = st.sidebar.multiselect("Drop Columns (optional)", df.columns, default=["id", "name", "compas_screening_date", "dob", "c_jail_in", "c_jail_out", label_col, protected_attr])
    features = df.drop(columns=drop_cols)
    labels = df[label_col]

    # Encode protected attribute if necessary
    if df[protected_attr].dtype == 'object':
        df[protected_attr] = df[protected_attr].astype('category').cat.codes
    privileged = int(df[protected_attr].value_counts().idxmax())
    unprivileged = [v for v in df[protected_attr].unique() if v != privileged][0]

    combined_df = pd.concat([features, labels, df[[protected_attr]]], axis=1)

    # impute missing values
    st.sidebar.subheader("🧹 Data Cleaning")
    # Combine features, label, and protected attribute into one DataFrame
    combined_df = pd.concat([features.copy(), labels.copy(), df[[protected_attr]].copy()], axis=1)

    st.sidebar.subheader("🧹 Data Cleaning")
    impute_strategy = st.sidebar.selectbox(
        "Missing Value Handling Strategy",
        ["Drop all rows with missing values", "Fill all missing with median", "Fill all missing with mode"]
    )

    # Track nulls before cleaning
    null_summary = combined_df.isnull().sum()
    null_cols = null_summary[null_summary > 0]

    if not null_cols.empty:
        st.warning(f"⚠️ Missing values found in: {', '.join(null_cols.index)}")
    else:
        st.success("✅ No missing values found in selected data.")

    # Apply chosen cleaning strategy
    clean_df = combined_df.copy()

    if impute_strategy == "Drop all rows with missing values":
        clean_df.dropna(inplace=True)
    elif impute_strategy == "Fill all missing with median":
        for col in clean_df.columns:
            if clean_df[col].isnull().any():
                if clean_df[col].dtype in [np.float64, np.int64]:
                    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
                else:
                    clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
    elif impute_strategy == "Fill all missing with mode":
        for col in clean_df.columns:
            if clean_df[col].isnull().any():
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

    # Final check
    if clean_df.isnull().any().any():
        st.error("❌ Some missing values could not be imputed.")
        st.stop()

    # Show summary of rows dropped or filled
    original_count = combined_df.shape[0]
    final_count = clean_df.shape[0]
    dropped_count = original_count - final_count

    if impute_strategy == "Drop all rows with missing values" and dropped_count > 0:
        st.info(f"ℹ️ Dropped {dropped_count} row(s) with missing values.")
    else:
        st.success(f"✅ Cleaned dataset: {final_count} rows × {clean_df.shape[1]} columns.")

    # Display cleaned data
    st.write("🧼 Cleaned DataFrame Preview")
    st.dataframe(clean_df.head())

    # Stop if empty
    if final_count == 0:
        st.error("🚨 No data left after cleaning. Try another strategy or check your dataset.")
        st.stop()

    # Final clean DataFrame
    st.write("✅ Cleaned DataFrame", clean_df.head())
    st.dataframe(clean_df.describe())
    
    # Stop the app if there's no data left
    if clean_df.shape[0] == 0:
        st.error("🚨 All rows were dropped due to missing values! Please clean your dataset or adjust selected columns.")
        st.stop()

    dataset = BinaryLabelDataset(df=clean_df,
                             label_names=[label_col],
                             protected_attribute_names=[protected_attr])


    # -------------------- Split --------------------
    train, test = dataset.split([0.7], shuffle=True)

    # -------------------- Reweighing --------------------
    use_reweighing = st.sidebar.checkbox("Apply Reweighing", value=True)
    if use_reweighing:
        RW = Reweighing(unprivileged_groups=[{protected_attr: unprivileged}],
                        privileged_groups=[{protected_attr: privileged}])
        train = RW.fit_transform(train)

    # -------------------- Model Training --------------------
    X_train = StandardScaler().fit_transform(train.features)
    y_train = train.labels.ravel()
    X_test = StandardScaler().fit_transform(test.features)
    y_test = test.labels.ravel()

    st.sidebar.subheader("🤖 Model Selection")
    model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_pred = test.copy()
    test_pred.labels = y_pred

    # -------------------- Metrics --------------------
    metric = ClassificationMetric(test, test_pred,
                                  unprivileged_groups=[{protected_attr: unprivileged}],
                                  privileged_groups=[{protected_attr: privileged}])

    st.subheader("📊 Fairness & Performance Metrics")
    acc = accuracy_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Disparate Impact", f"{metric.disparate_impact():.3f}")
    col3.metric("Equal Opportunity Diff", f"{metric.equal_opportunity_difference():.3f}")

    fpr_u = metric.false_positive_rate(False)
    fpr_p = metric.false_positive_rate(True)
    st.metric("False Positive Rate (Unprivileged)", round(fpr_u, 3))
    st.metric("False Positive Rate (Privileged)", round(fpr_p, 3))

    # -------------------- Plots --------------------
    st.subheader("📈 Visualizations")
    with st.expander("🔍 Histogram: Predicted vs Actual"):
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(y_test, color="blue", label="Actual", kde=False)
        sns.histplot(y_pred, color="red", label="Predicted", kde=False)
        plt.legend()
        st.pyplot(fig)

    with st.expander("📊 Bar Plot: decile_score / v_decile_score"):
        if 'decile_score' in df.columns:
            fig2 = plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='decile_score', hue='race')
            st.pyplot(fig2)
        if 'v_decile_score' in df.columns:
            fig3 = plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='v_decile_score', hue='race')
            st.pyplot(fig3)

    with st.expander("📈 score_text Distribution"):
        if 'score_text' in df.columns:
            fig4 = plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x='score_text', hue='race')
            st.pyplot(fig4)

    # -------------------- PDF Export --------------------
    st.subheader("📄 Export Report as PDF")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"COMPAS Bias Report\\nModel: {model_choice}\\n"
                                  f"Protected Attribute: {protected_attr}\\n"
                                  f"Disparate Impact: {metric.disparate_impact():.3f}\\n"
                                  f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.3f}\\n"
                                  f"FPR (Unprivileged): {fpr_u:.3f}\\n"
                                  f"FPR (Privileged): {fpr_p:.3f}\\n"
                                  f"Accuracy: {acc:.2%}")
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button("Download Report", data=buffer.getvalue(), file_name="compas_bias_report.pdf")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + AIF360")

else:
    st.info("👈 Upload `compas_scores.csv` from your machine to get started.")
