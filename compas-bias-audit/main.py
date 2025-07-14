import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
import zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from aif360.datasets import CompasDataset, BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from fpdf import FPDF
import janitor

# Page setup
st.set_page_config(page_title="COMPAS Bias Audit", layout="wide")
st.title("🔍 COMPAS Recidivism Bias Audit")

@st.cache_data
def load_default_dataset():
    return CompasDataset(protected_attribute_names=['race'],
                         privileged_classes=[['Caucasian']],
                         features_to_drop=['sex'])

def convert_to_aif360(df, protected_attr, label_col='two_year_recid'):
    features = df.drop(columns=[label_col])
    labels = df[label_col]
    return BinaryLabelDataset(df=pd.concat([features, labels], axis=1),
                              label_names=[label_col],
                              protected_attribute_names=[protected_attr])

uploaded_file = st.sidebar.file_uploader("Upload custom dataset (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded")
    st.dataframe(df.head())
    label_col = st.sidebar.selectbox("Select label column", df.columns)
    protected_attr = st.sidebar.selectbox("Select protected attribute", df.columns)
    dataset = convert_to_aif360(df, protected_attr, label_col)
else:
    dataset = load_default_dataset()
    protected_attr = 'race'
    label_col = 'two_year_recid'

train, test = dataset.split([0.7], shuffle=True)

use_reweighing = st.sidebar.checkbox("Apply Reweighing", value=True)
if use_reweighing:
    RW = Reweighing(unprivileged_groups=[{protected_attr: 1}], privileged_groups=[{protected_attr: 0}])
    train = RW.fit_transform(train)

scaler = StandardScaler()
X_train = scaler.fit_transform(train.features)
y_train = train.labels.ravel()
X_test = scaler.transform(test.features)
y_test = test.labels.ravel()

model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_pred = test.copy()
test_pred.labels = y_pred

metric = ClassificationMetric(test, test_pred,
                              unprivileged_groups=[{protected_attr: 1}],
                              privileged_groups=[{protected_attr: 0}])

fpr_unpriv = metric.false_positive_rate(False)
fpr_priv = metric.false_positive_rate(True)
eq_diff = metric.equal_opportunity_difference()
di = metric.disparate_impact()
acc = accuracy_score(y_test, y_pred)

st.subheader("📊 Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("FPR (Unprivileged)", round(fpr_unpriv, 3))
col2.metric("FPR (Privileged)", round(fpr_priv, 3))
col3.metric("Disparate Impact", round(di, 3))
st.metric("Accuracy", f"{acc:.2%}")

st.subheader("📈 Visualizations")
fig1 = plt.figure()
sns.barplot(x=['Unprivileged', 'Privileged'], y=[fpr_unpriv, fpr_priv])
plt.title("False Positive Rate by Group")
st.pyplot(fig1)

fig2 = plt.figure()
sns.histplot(y_test, color='blue', label='Actual', binwidth=1)
sns.histplot(y_pred, color='red', label='Predicted', binwidth=1)
plt.legend()
plt.title("Predicted vs. Actual")
st.pyplot(fig2)

st.subheader("📄 Export Report as PDF")
if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="COMPAS Fairness Audit Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Model: {model_name}\\n"
                              f"Protected Attribute: {protected_attr}\\n"
                              f"False Positive Rate (Unprivileged): {fpr_unpriv:.3f}\\n"
                              f"False Positive Rate (Privileged): {fpr_priv:.3f}\\n"
                              f"Equal Opportunity Difference: {eq_diff:.3f}\\n"
                              f"Disparate Impact: {di:.3f}\\n"
                              f"Accuracy: {acc:.2%}")
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    st.download_button("Download PDF", pdf_output.getvalue(), file_name="fairness_report.pdf")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and AIF360")
