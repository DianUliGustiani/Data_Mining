import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.frequent_patterns import fpgrowth, association_rules

# URL Google Drive
url = 'https://drive.google.com/file/d/1HiJwa4ZOyUWGn20LtrH_FdFzxZWWT-Ss/view?usp=sharing'

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(url)

df_encoded = load_data()

# Apply FP-Growth Algorithm
frequent_itemsets = fpgrowth(df_encoded, min_support=0.007, use_colnames=True)

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
