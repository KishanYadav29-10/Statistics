import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to detect outliers using Z-score
def detect_outliers_zscore(data, threshold=1.5):
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))

st.title("Outlier Detection & Treatment Tool")
st.write("Upload a CSV file to detect and visualize outliers using Z-score and IQR methods.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of the Dataset:", df.head())
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found in the dataset.")
    else:
        feature = st.selectbox("Select a numeric feature for outlier detection", numeric_columns)
        
        # Detect outliers
        df['Z-Score Outlier'] = detect_outliers_zscore(df[feature])
        df['IQR Outlier'] = detect_outliers_iqr(df[feature])
        
        st.write("### Outlier Summary:")
        st.write(df[['Z-Score Outlier', 'IQR Outlier']].value_counts())
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(y=df[feature], ax=axes[0])
        axes[0].set_title("Boxplot of " + feature)
        
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Distribution of " + feature)
        
        st.pyplot(fig)
        
        # Option to remove outliers
        remove_outliers = st.checkbox("Remove Detected Outliers")
        if remove_outliers:
            df_cleaned = df[~df['IQR Outlier']]
            st.write("### Cleaned Data Sample:", df_cleaned.head())
            st.download_button("Download Cleaned Data", df_cleaned.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")
