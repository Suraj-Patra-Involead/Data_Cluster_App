import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import io,base64,tempfile
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN,KMeans
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes
from lamma_summary import give_summary
from cluster_summury import summarize_clusters

st.set_page_config(page_title="Clustering Assistance AI",page_icon="☀️",layout="wide")

# Page Management
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []

# Cluster Data Upload and summury
if st.session_state.page == "upload":
    st.title(":blue[📊 Upload Your Dataset :]",width="stretch")
    uploaded_file = st.file_uploader("Upload a CSV or Excel File: ",type=['csv','excel'])
    
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            st.markdown(":red[Upload a valid document]")
            st.stop()
        st.session_state.df = df
    
        st.subheader(":green[Dataset Preview]")
        if st.button(':blue[Expand 🦣]'):
            st.dataframe(df)
        if st.button(":blue[Small] 🤏"):
            st.dataframe(df.head())
        st.markdown("### 📌 Dataset Diagnostics", unsafe_allow_html=True)
        st.markdown(f"**Shape:** {df.shape[0]} Rows and {df.shape[1]} Columns")
        st.markdown(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.markdown(f"**Duplicates:** {df.duplicated().sum()}")

        if st.button(":blue[More Info: ]"):
            st.markdown(f"**Null Values in each column:**")
            st.dataframe(df.isnull().sum())
            st.markdown(f"**Numerical Attributes:**")
            st.dataframe(df.select_dtypes(include=['int','float']).describe())
            st.markdown(f"**Categorical Attributes:**")
            st.dataframe(df.select_dtypes(include=['object']).describe())
        if st.button(":blue[Close: ]"):
            pass
        
        col1 , col2 = st.columns(2)

        with col1:
            if st.button(":blue[📄 Generate Llama3 Summary]"):
                num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                nulls_summary = df.isnull().sum().to_dict()
                duplicates_count = int(df.duplicated().sum())
                short_num_cols = num_cols[:5] + (["..."] if len(num_cols) > 5 else [])
                short_cat_cols = cat_cols[:5] + (["..."] if len(cat_cols) > 5 else [])
                short_nulls = dict(list(nulls_summary.items())[:5])

                summary_text = give_summary(num_cols,cat_cols,nulls_summary,duplicates_count,short_cat_cols,short_num_cols,short_nulls)

                st.markdown("### 📝 Llama3 Summary")

                st.text_area("Summary Preview", summary_text, height=250)

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                def safe_print_to_pdf(text):
                        lines = text.split("\n")
                        for line in lines:
                            words = line.split(" ")
                            current_line = ""
                            for word in words:
                                if len(word) > 80:
                                    word = word[:75] + "..."
                                if len(current_line + " " + word) <= 90:
                                    current_line += " " + word
                                else:
                                    pdf.cell(0, 10, current_line.strip(), ln=True)
                                    current_line = word
                            if current_line.strip():
                                pdf.cell(0, 10, current_line.strip(), ln=True)

                safe_print_to_pdf(summary_text)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        pdf.output(tmp.name)
                        with open(tmp.name, "rb") as f:
                            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
                
                st.markdown("### 📄 Download PDF Summary")
                st.markdown(
                        f'<a href="data:application/pdf;base64,{base64_pdf}" download="dataset_summary.pdf">📥 Download PDF</a>',
                        unsafe_allow_html=True
                )
        with col2:
            if st.button("🔢📊:blue[Go For Clustering]"):
                st.session_state.page = "advanced"
                st.rerun()

if st.session_state.page == "advanced":
    st.title("🔢:blue[Select Clustering Techniques You wnat to Perform]")
    st.write("🛠️:green[Select Clustering Method]")
    df = st.session_state.df.drop_duplicates().copy()
    numerical = df.select_dtypes(['int', 'float']).columns.to_list()
    categorical = df.select_dtypes('object').columns.to_list()
    imputer = SimpleImputer(strategy="most_frequent")
    df[df.columns] = imputer.fit_transform(df)
    df[numerical] = df[numerical].astype('float')
    df[categorical] = df[categorical].astype('object')
    st.session_state.df_clean = df.copy()

    method = st.radio("",['K-Means','GMM','DBSCAN','K-Prototype'])

    if st.button("⬅️ Back to Upload"):
        st.session_state.page = "upload"
        st.rerun()

    if st.button("Proceed to Feature Selection"):
        st.session_state.selected_method = method
        st.session_state.page = "preprocess_and_cluster"
        st.rerun()
    
elif st.session_state.page == "preprocess_and_cluster":
    st.title("🔍 :blue[Feature Selection and Transformation]")

    df_clean = st.session_state.df_clean.copy()
    method = st.session_state.selected_method

    all_cols = df_clean.columns.tolist()

    selected = st.multiselect("Select features for clustering:", all_cols, default=st.session_state.selected_features)
    st.session_state.selected_features = selected

    if st.button("⬅️ Back to Method Selection"):
      st.session_state.page = "advanced"
      st.rerun()
    
    if selected:
        # Keep original data for summary
        original_df = st.session_state.df_clean[selected].copy()
        df = original_df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist() 

        if method in ['K-Means','GMM','DBSCAN','K-Prototype'] and cat_cols:
            encoding_option = st.selectbox("Encoding method for categorical data:", ["Label Encoding", "One-Hot Encoding"])
        
        scaling_option = st.selectbox("Scaling method for numeric data:", ["StandardScaler", "MinMaxScaler"])
        st.session_state.encoding_option = encoding_option

        # Apply encoding
        if method in ['K-Means','GMM','DBSCAN','K-Prototype'] and cat_cols:
            if encoding_option == "Label Encoding":
                for col in cat_cols:
                    df[col] = LabelEncoder().fit_transform(df[col])
            elif encoding_option == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=cat_cols)
                cat_cols = []  # All converted
            
        # Apply scaling
        if num_cols:
            scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        
        st.session_state.df_clean_featured = df.copy()

        if st.button(":blue[Preview Dataset After Feature Engineering]"):
            st.dataframe(df.head())
        if st.button(":blue[Go For Clusering Table]"):
            st.session_state.page = "clustering"
        if st.button(":blue[Close]"):
            pass

elif st.session_state.page == "clustering":
    st.title("🔢:blue[Clustering Summary Generation]")

    if st.button("⬅️ :blue[Back to Feature Selection and Transformation]"):
      st.session_state.page = "preprocess_and_cluster"
      st.rerun()
    
    df = st.session_state.df_clean_featured.copy()
    original_df = st.session_state.df_clean.copy()

    num_cols = original_df.select_dtypes(include=['int','float']).columns.to_list()
    # cat_cols = original_df.select_dtypes('object').columns.to_list()
    try:
        if st.session_state.selected_method == "K-Means":
                k = st.slider("Select Number of clusters (k):", 2, 10, 3)
                model = KMeans(n_clusters=k, random_state=42)
                clusters = model.fit_predict(df)
                original_df['Cluster'] = clusters

        elif st.session_state.selected_method == "GMM":
                k = st.slider("Select Number of components:", 2, 10, 3)
                model = GaussianMixture(n_components=k, random_state=42)
                clusters = model.fit_predict(df)
                original_df['Cluster'] = clusters

        elif st.session_state.selected_method == "DBSCAN":
                eps = st.slider("Select Epsilon:", 0.1, 5.0, 0.5)
                min_samples = st.slider("Select Minimum Samples:", 2, 10, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(df)
                original_df['Cluster'] = clusters

        elif st.session_state.selected_method == "K-Prototype":
            k = st.slider("Number of clusters (k):", 2, 10, 3)
            # Update num_cols to only include numerical columns present in df
            num_cols = df.select_dtypes(include=['int', 'float']).columns.to_list()
            # Use selected_features to identify categorical columns before encoding
            cat_cols = [col for col in st.session_state.selected_features if col in original_df.columns and original_df[col].dtype in ['object', 'category']]
            # Adjust cat_cols based on encoding applied in preprocessing
            if 'encoding_option' in locals() and st.session_state.encoding_option == "One-Hot Encoding":
                cat_cols = []  # One-Hot Encoding removes original categorical columns
            elif 'encoding_option' in locals() and st.session_state.encoding_option == "Label Encoding":
                cat_cols = [col for col in cat_cols if col in df.columns]  # Only keep columns still in df
            for col in num_cols:
                df[col] = MinMaxScaler().fit_transform(df[[col]])
            matrix = df.to_numpy()
            cat_idx = [df.columns.get_loc(col) for col in cat_cols if col in df.columns]
            model = KPrototypes(n_clusters=k, init='Cao', verbose=0, random_state=42)
            clusters = model.fit_predict(matrix, categorical=cat_idx)
            original_df['Cluster'] = clusters
        
        st.session_state.clustered_df = original_df
        st.success("✅ Clustering completed.")
        if st.button(":blue[Expand View 🦣]"):
            st.dataframe(original_df)
        if st.button(":blue[Small View 🤏]"):
            st.dataframe(original_df.head())
        if st.button(":blue[Close 📕]"):
            pass

        selected = st.session_state.selected_features.copy()
        with st.expander("📋 Cluster Summary Table Preferences"):
                summary_preferences = {}
                for col in selected:
                    col_type = original_df[col].dtype
                    if col_type in ['int64', 'float64']:
                        options = ['mean', 'median', 'mode', 'min', 'max', 'sum']
                    else:
                        options = ['count', 'percentage']
                    summary_preferences[col] = st.selectbox(f"Summary method for {col}", options, key=col)
        
        # Custom cluster names
        unique_clusters = sorted(original_df['Cluster'].unique())
        cluster_names = {}
        st.subheader("Custom Cluster Names")
        for i, c in enumerate(unique_clusters):
            default_name = f"Cluster {c}"
            cluster_names[c] = st.text_input(f"Name for Cluster {c}",default_name, key=f"cluster_name_{c}") 
        if st.button("📊 Generate Cluster Summary Table"):
            summary_df = summarize_clusters(original_df, 'Cluster', summary_preferences, cluster_names)
            st.dataframe(summary_df)
    except Exception as e:
        st.write(e)