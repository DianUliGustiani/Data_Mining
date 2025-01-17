import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from zipfile import ZipFile
from io import BytesIO
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# ID Google Drive dan nama file output
file_id = '1eM_QYD6MSu7-heV46186xxtkNXoEU49c'

# Load dataset
@st.cache_data
def load_data(file_id):
    try:
        st.write(f"Downloading data from Google Drive file ID: {file_id}")
        output_zip = 'dataset.zip'
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=output_zip, unzip=False)

        if os.path.exists(output_zip):
            st.write("Extracting dataset from zip file")
            with ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall('data/')

            # Assuming your dataset is CSV inside the zip file
            csv_file = 'data/dataset.csv'
            df = pd.read_csv(csv_file)
            st.write(f"Data loaded successfully with shape {df.shape}")
            return df
        else:
            st.error("Failed to download the file.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df_encoded = load_data(file_id)

if df_encoded is not None and not df_encoded.empty:
    try:
        # Continue with your data analysis and visualization
        st.write("Applying FP-Growth algorithm")
        # Apply FP-Growth Algorithm
        frequent_itemsets = fpgrowth(df_encoded, min_support=0.007, use_colnames=True)

        if frequent_itemsets.empty:
            st.error("No frequent itemsets found. Please adjust the minimum support or check the data.")
        else:
            st.write("Generating association rules")
            # Generate the association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

            # Convert frozensets to lists for serialization
            rules['antecedents'] = rules['antecedents'].apply(list)
            rules['consequents'] = rules['consequents'].apply(list)

            # Streamlit app
            st.title('Market Basket Analysis')
            st.write('Aplikasi ini menampilkan hasil analisis market basket menggunakan algoritma FP-Growth.')

            st.header('Data Transaksi')
            st.write('Tabel data transaksi yang telah di-encode:')
            st.dataframe(df_encoded.head())

            st.header('Frequent Itemsets')
            st.write('Itemset yang sering muncul berdasarkan min_support:')
            st.dataframe(frequent_itemsets)

            st.header('Association Rules')
            st.write('Aturan asosiasi yang dihasilkan:')
            st.dataframe(rules)

            st.header('Scatter Plot: Support vs. Confidence')
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x="support", y="confidence", size="lift", data=rules, hue="lift", palette="viridis", sizes=(20, 200), ax=ax)
            plt.title('Market Basket Analysis - Support vs. Confidence (Size = Lift)')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            st.pyplot(fig)

            st.header('Interactive Scatter Plot')
            fig = px.scatter(rules, x="support", y="confidence", size="lift",
                             color="lift", hover_name="consequents",
                             title='Market Basket Analysis - Support vs. Confidence',
                             labels={'support': 'Support', 'confidence': 'Confidence'})
            fig.update_layout(
                xaxis_title='Support',
                yaxis_title='Confidence',
                coloraxis_colorbar_title='Lift',
                showlegend=True
            )
            st.plotly_chart(fig)
    except ZeroDivisionError:
        st.error("Encountered a division by zero error. Please check the data and the min_support value.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Failed to load data or DataFrame is empty. Please check the CSV file.")
