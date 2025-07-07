# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation & Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("datanasabah.csv")

data = load_data()

# Mapping kategori
data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-Laki': 1, 'Perempuan': 2})
data['jenis_produk'] = data['jenis_produk'].map({'tabungan': 1, 'kartu_kredit': 2, 'deposito': 3})
data['pengguna_mobile_banking'] = data['pengguna_mobile_banking'].map({'YA': 1, 'TIDAK': 2})

# Sidebar filter
with st.sidebar:
    st.header("🔎 Filter")
    selected_produk = st.multiselect("Jenis Produk", options=data['jenis_produk'].unique(), default=data['jenis_produk'].unique())
    filtered_data = data[data['jenis_produk'].isin(selected_produk)]

st.subheader("📌 Ringkasan Data")
st.dataframe(filtered_data.head())

st.markdown("### 📈 Korelasi antar Fitur Numerik")
numerical_cols = ['umur', 'pendapatan', 'saldo_rata_rata', 'jumlah_transaksi', 'frekuensi_kunjungi_cabang', 'skor_kredit']
corr = filtered_data[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, ax=ax)
st.pyplot(fig)
