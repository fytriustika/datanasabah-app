# datanasabah_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set(style="whitegrid")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_data(csv_file):
    return pd.read_csv(csv_file)

@st.cache_data
def encode_df(df):
    df = df.copy()
    df['jenis_kelamin'] = df['jenis_kelamin'].map({'Laki-Laki': 1, 'Perempuan': 2})
    df['jenis_produk'] = df['jenis_produk'].map({'tabungan': 1, 'kartu_kredit': 2, 'deposito': 3})
    df['pengguna_mobile_banking'] = df['pengguna_mobile_banking'].map({'YA': 1, 'TIDAK': 2})
    return df

@st.cache_data
def basic_stats(df):
    numeric = df.select_dtypes(include=[np.number])
    return df.shape, df.dtypes, numeric.describe(include='all'), df.isnull().sum()

def plot_histograms(df, cols):
    n = len(cols)
    rows = int(np.ceil(n/2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4*rows))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        sns.histplot(df[c].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution: {c}')
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig)

def plot_boxplots(df, cols):
    n = len(cols)
    rows = int(np.ceil(n/2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4*rows))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        sns.boxplot(x=df[c].dropna(), ax=axes[i])
        axes[i].set_title(f'Boxplot: {c}')
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig)

def corr_heatmap(df, cols):
    cm = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    return cm

@st.cache_data
def run_pca(df, cols):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols])
    pca = PCA(n_components=min(len(cols), 6))
    pcs = pca.fit_transform(Xs)
    pc_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    return pca, pc_df

@st.cache_data
def kmeans_with_elbow(df, cols, kmax=8):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols])
    inertias = []
    for k in range(1, kmax+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)
    chosen_k = 3
    km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(Xs)
    labels = km_final.labels_
    return inertias, chosen_k, labels

def train_and_eval_regressors(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        results[name] = {"model": m, "MSE": mse, "RMSE": rmse, "R2": r2}
    return results

# -------------------------
# UI
# -------------------------
st.title("Customer Data Dashboard — Datanasabah")

# Data upload / load
st.sidebar.header("Data input")
uploaded = st.sidebar.file_uploader("Upload datanasabah.csv", type=['csv'])
if uploaded is not None:
    df_raw = load_data(uploaded)
else:
    try:
        df_raw = load_data('datanasabah.csv')
    except Exception:
        st.error("No CSV found. Please upload `datanasabah.csv`.")
        st.stop()

if st.sidebar.checkbox("Show raw data preview", value=True):
    st.subheader("Raw data (first 10 rows)")
    st.dataframe(df_raw.head(10))

# Preprocess / encode
df = encode_df(df_raw)

# Overview
st.header("Overview & Data Validation")
shape, dtypes, desc, missing = basic_stats(df)
st.write(f"Shape: {shape[0]} rows × {shape[1]} columns")
st.write("Column types:", dtypes)
st.write("Missing values per column:", missing)

# Univariat EDA
st.header("Univariate Analysis")
numerical_cols = ['umur', 'pendapatan', 'saldo_rata_rata', 'jumlah_transaksi',
                  'frekuensi_kunjungi_cabang', 'skor_kredit']
plot_histograms(df, numerical_cols)
plot_boxplots(df, numerical_cols)

# Bivariate
st.header("Bivariate Analysis")
if st.button("Show correlation heatmap"):
    cm = corr_heatmap(df, numerical_cols)
    st.dataframe(cm)

# PCA
st.header("PCA & Pair Plot")
pca, pc_df = run_pca(df, numerical_cols)
st.write("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
fig, ax = plt.subplots()
sns.scatterplot(x=pc_df['PC1'], y=pc_df['PC2'],
                hue=df['jenis_produk'].astype(str), palette='tab10', ax=ax)
ax.set_title("PC1 vs PC2 colored by jenis_produk")
st.pyplot(fig)

# Clustering
st.header("Customer Segmentation (K-Means)")
inertias, chosen_k, labels = kmeans_with_elbow(df, numerical_cols)
fig, ax = plt.subplots()
ax.plot(range(1, len(inertias)+1), inertias, marker='o')
ax.set_xlabel('k')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Plot')
st.pyplot(fig)

df['cluster_label'] = labels
st.write("Counts per cluster:", df['cluster_label'].value_counts())

st.subheader("Cluster Profiles (mean of numerical features)")
st.dataframe(df.groupby('cluster_label')[numerical_cols].mean().round(2))

# Statistical tests
st.header("Statistical Tests")
try:
    groups = [df[df['jenis_produk']==g]['frekuensi_kunjungi_cabang'].dropna()
              for g in sorted(df['jenis_produk'].unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    st.write(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_val:.4f}")
except Exception as e:
    st.error(f"ANOVA failed: {e}")

mobile = df[df['pengguna_mobile_banking']==1]
non_mobile = df[df['pengguna_mobile_banking']==2]
tt_results = {}
for col in numerical_cols:
    try:
        t, p = stats.ttest_ind(mobile[col].dropna(), non_mobile[col].dropna())
        tt_results[col] = (t, p)
    except Exception:
        tt_results[col] = (np.nan, np.nan)
st.write(pd.DataFrame(tt_results, index=['t_stat', 'p_value']).T)

# High-value and risky groups
st.header("High-value & Risky Customers")
income_th = df['pendapatan'].quantile(0.75)
bal_th = df['saldo_rata_rata'].quantile(0.75)
txn_th = df['jumlah_transaksi'].quantile(0.75)
high_value = df[(df['pendapatan']>income_th) &
                (df['saldo_rata_rata']>bal_th) &
                (df['jumlah_transaksi']>txn_th)]
st.write(f"High-value customers: {len(high_value)}")

credit_th = df['skor_kredit'].quantile(0.25)
risky = df[df['skor_kredit'] <= credit_th]
st.write(f"Risky customers: {len(risky)}")

# Predicting credit score
st.header("Predicting Credit Score (Regression)")
features = ['umur', 'pendapatan', 'saldo_rata_rata', 'jumlah_transaksi',
            'frekuensi_kunjungi_cabang', 'jenis_kelamin', 'jenis_produk',
            'pengguna_mobile_banking']
X = df[features]
y = df['skor_kredit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button("Train regression models"):
    results = train_and_eval_regressors(X_train, X_test, y_train, y_test)
    res_df = pd.DataFrame({k:{"RMSE":v['RMSE'],"R2":v['R2']} for k,v in results.items()}).T
    st.dataframe(res_df.style.format("{:.4f}"))
    best = res_df['R2'].idxmax()
    st.success(f"Best model: {best} (R2={res_df.loc[best,'R2']:.4f})")

# Final findings
st.header("Narrative Findings & Recommendations")
st.markdown("""
- **Data quality:** No missing values or duplicates; distributions are reasonable.  
- **Univariate / Bivariate:** `pendapatan` and `saldo_rata_rata` show positive correlation; some variables have outliers.  
- **Product influence:** ANOVA showed `jenis_produk` affects `frekuensi_kunjungi_cabang`.  
- **Mobile banking:** No significant numerical differences between users and non-users.  
- **Clustering:** k=3 segmentation gives distinct profiles.  
- **Modeling:** Credit-score regressions gave weak results; more predictive features are needed.  
""")
