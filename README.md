# Data Exploration and Customer Analysis Project

**Project Overview:**

This project involved a comprehensive data exploration and analysis of a bank customer dataset to gain a deep understanding of the customer base, identify distinct customer segments, analyze the relationships between various demographic and behavioral factors, and uncover insights that could inform targeted business strategies, such as customer segmentation, risk assessment, and identifying cross-selling opportunities.

**Project Details:**

*   **Data Information:**
    *   Dataset contains customer data from a bank.
    *   Includes demographic (age, gender), financial behavior (income, average balance, transaction count, branch visit frequency, mobile banking usage, credit score), and product information.
    *   Dataset size: 100 observations and 10 variables.

*   **Data Description:**
    *   Numerical variables (`umur`, `pendapatan`, `saldo_rata_rata`, `jumlah_transaksi`, `frekuensi_kunjungi_cabang`, `skor_kredit`) have varying distributions, with some indicating potential outliers.
    *   Categorical variables (`jenis_kelamin`, `jenis_produk`, `pengguna_mobile_banking`) show the distribution of customers across different categories.
    *   `nasabah_id` is a unique identifier.

*   **EDA (Exploratory Data Analysis):**
    *   **Univariate Analysis:** Examined the distribution of each variable using descriptive statistics, histograms, box plots, and count plots.
    *   **Bivariate Analysis:** Explored relationships between pairs of variables through scatter plots, box plots (categorical vs. numerical), and correlation matrix for numerical variables.
    *   **Multivariate Analysis (PCA):** Used PCA to visualize data in lower dimensions and observe potential groupings.
    *   **Customer Segmentation (K-Means):** Applied K-Means clustering to identify customer segments based on numerical features. The Elbow method was used to inform the number of clusters.
    *   **Categorical Influence Analysis:** Performed T-tests and ANOVA to test the significance of differences in numerical variable means across categorical groups.
    *   **Outlier Analysis:** Identified potential outliers in numerical variables using the IQR method.
    *   **Specific Relationship Analysis:** Focused analysis on the relationship between product type and branch visit frequency, and the comparison between mobile banking users and non-users.
    *   **Factors Influencing Credit Score:** Explored relationships between features and credit score through correlation, grouped analysis, and box plots.

*   **Data Cleaning and Preprocessing:**
    *   Checked for and confirmed the absence of missing values.
    *   Checked for and confirmed the absence of duplicate rows.
    *   Encoded categorical variables (`jenis_kelamin`, `jenis_produk`, `pengguna_mobile_banking`) into numerical representations (1 and 2 or 1, 2, 3) for use in quantitative analysis and modeling.
    *   Standardized numerical data for clustering.

*   **Feature Selection:**
    *   For clustering, all numerical variables were initially used.
    *   For the credit score prediction modeling scenario, all features except 'nasabah_id' and 'skor_kredit' (the target variable) were selected as features.

*   **Model Training and Evaluation (Predicting Credit Score - Regression Scenario):**
    *   The task was framed as a regression problem to predict 'skor\_kredit'.
    *   Data was split into training (80%) and testing (20%) sets.
    *   Selected regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor) were trained on the training data.
    *   Models were evaluated on the testing data using MSE, RMSE, and R2 Score.

*   **Model Evaluation Results on Validation Set:**
    *   Performance metrics for the tested regression models were:
        *   Random Forest Regressor: MSE = 22987.65, RMSE = 151.6168, R2 Score = -0.2035
        *   Linear Regression: MSE = 23239.73, RMSE = 152.4458, R2 Score = -0.2167
        *   Decision Tree Regressor: MSE = 34500.00, RMSE = 185.7418, R2 Score = -0.8063
        *   Gradient Boosting Regressor: MSE = 34592.96, RMSE = 185.9918, R2 Score = -0.8111
    *   All models resulted in negative R2 scores, indicating poor predictive performance on this dataset with the selected features.
    *   Random Forest Regressor performed best among the tested models based on R2 score, despite the negative value.

# Data Exploration and Customer Analysis Project

**Project Overview:**

This project involved a comprehensive data exploration and analysis of a bank customer dataset to gain a deep understanding of the customer base, identify distinct customer segments, analyze the relationships between various demographic and behavioral factors, and uncover insights that could inform targeted business strategies, such as customer segmentation, risk assessment, and identifying cross-selling opportunities.

**Project Details:**

*   **Data Information:**
    *   Dataset contains customer data from a bank.
    *   Includes demographic (age, gender), financial behavior (income, average balance, transaction count, branch visit frequency, mobile banking usage, credit score), and product information.
    *   Dataset size: 100 observations and 10 variables.

*   **Data Description:**
    *   Numerical variables (`umur`, `pendapatan`, `saldo_rata_rata`, `jumlah_transaksi`, `frekuensi_kunjungi_cabang`, `skor_kredit`) have varying distributions, with some indicating potential outliers.
    *   Categorical variables (`jenis_kelamin`, `jenis_produk`, `pengguna_mobile_banking`) show the distribution of customers across different categories.
    *   `nasabah_id` is a unique identifier.

*   **EDA (Exploratory Data Analysis):**
    *   **Univariate Analysis:** Examined the distribution of each variable using descriptive statistics, histograms, box plots, and count plots.
    *   **Bivariate Analysis:** Explored relationships between pairs of variables through scatter plots, box plots (categorical vs. numerical), and correlation matrix for numerical variables.
    *   **Multivariate Analysis (PCA):** Used PCA to visualize data in lower dimensions and observe potential groupings.
    *   **Customer Segmentation (K-Means):** Applied K-Means clustering to identify customer segments based on numerical features. The Elbow method was used to inform the number of clusters.
    *   **Categorical Influence Analysis:** Performed T-tests and ANOVA to test the significance of differences in numerical variable means across categorical groups.
    *   **Outlier Analysis:** Identified potential outliers in numerical variables using the IQR method.
    *   **Specific Relationship Analysis:** Focused analysis on the relationship between product type and branch visit frequency, and the comparison between mobile banking users and non-users.
    *   **Factors Influencing Credit Score:** Explored relationships between features and credit score through correlation, grouped analysis, and box plots.

*   **Data Cleaning and Preprocessing:**
    *   Checked for and confirmed the absence of missing values.
    *   Checked for and confirmed the absence of duplicate rows.
    *   Encoded categorical variables (`jenis_kelamin`, `jenis_produk`, `pengguna_mobile_banking`) into numerical representations (1 and 2 or 1, 2, 3) for use in quantitative analysis and modeling.
    *   Standardized numerical data for clustering.

*   **Feature Selection:**
    *   For clustering, all numerical variables were initially used.
    *   For the credit score prediction modeling scenario, all features except 'nasabah_id' and 'skor_kredit' (the target variable) were selected as features.

*   **Model Training and Evaluation (Predicting Credit Score - Regression Scenario):**
    *   The task was framed as a regression problem to predict 'skor\_kredit'.
    *   Data was split into training (80%) and testing (20%) sets.
    *   Selected regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor) were trained on the training data.
    *   Models were evaluated on the testing data using MSE, RMSE, and R2 Score.

*   **Model Evaluation Results on Validation Set:**
    *   Performance metrics for the tested regression models were:
        *   Random Forest Regressor: MSE = 22987.65, RMSE = 151.6168, R2 Score = -0.2035
        *   Linear Regression: MSE = 23239.73, RMSE = 152.4458, R2 Score = -0.2167
        *   Decision Tree Regressor: MSE = 34500.00, RMSE = 185.7418, R2 Score = -0.8063
        *   Gradient Boosting Regressor: MSE = 34592.96, RMSE = 185.9918, R2 Score = -0.8111
    *   All models resulted in negative R2 scores, indicating poor predictive performance on this dataset with the selected features.
    *   Random Forest Regressor performed best among the tested models based on R2 score, despite the negative value.

*   **Other Findings:**
    *   **Customer Segmentation:** Identified 3 distinct clusters with varying characteristics (e.g., Cluster 1 with high branch visits and higher credit scores, Cluster 0 with low branch visits and lower credit scores).
    *   **Financial Behavior Relationships:** Moderate positive correlation between income and average balance. Weak correlations between other financial behaviors.
    *   **Product Influence on Behavior:** Product type significantly impacts branch visit frequency, with deposit holders visiting more often.
    *   **Mobile Banking Usage:** No statistically significant differences in numerical behaviors between mobile banking users and non-users were found.
    *   **Credit Score Factors:** Linear correlation is weak, but analysis suggests potential non-linear relationships or influence of other factors on credit score based on feature distributions across score ranges.
    *   **High-Value Customers:** Identified a small group based on high income, balance, and transactions.
    *   **Risky Customers:** Identified a group with lower credit scores.
    *   **Cross-selling/Up-selling Opportunities:** Identified potential opportunities based on segment profiles, product ownership, and customer characteristics (e.g., digital services for less engaged segments, investment products for high-balance segments, financial literacy for risky customers).

**Conclusion:**

This project involved a thorough exploration of the customer dataset, revealing key insights into customer profiles, behavior patterns, segmentation, and relationships between variables. While the predictive modeling for credit score faced challenges with the current features, the comprehensive EDA and analysis of specific relationships provided valuable, actionable insights for targeted marketing, product development, and risk management strategies.

**Conclusion:**

This project involved a thorough exploration of the customer dataset, revealing key insights into customer profiles, behavior patterns, segmentation, and relationships between variables. While the predictive modeling for credit score faced challenges with the current features, the comprehensive EDA and analysis of specific relationships provided valuable, actionable insights for targeted marketing, product development, and risk management strategies.
