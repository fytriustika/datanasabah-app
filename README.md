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

## Summary of Key Findings from Customer Data Exploration

Here is a summary of the main findings obtained from the in-depth exploration of the customer dataset:

### 1. General Customer Characteristics:
- The dataset consists of 100 customers with 10 features.
- The data is of good quality, with no missing values or duplicates.
- Customer age distribution varies, with most being in the productive to middle-aged range.
- Income and average balance distributions show diversity, with some potential outliers at higher values.
- Transaction count and branch visit frequency also vary among customers.
- Customer credit scores are distributed into several discrete categories (500, 600, 700, 800, 900).
- The proportion of male and female customers is relatively balanced.
- The most common products owned are credit cards and deposits, followed by savings.
- The number of mobile banking users and non-users is relatively balanced.

### 2. Customer Segmentation:
- Through K-Means clustering (with k=3), customers can be divided into several distinct segments based on their attributes and behaviors.
- **Cluster 0:** Tend to have low branch visit frequency, moderate income and average balance, and lower credit scores.
- **Cluster 1:** Tend to have high branch visit frequency, lower income and average balance compared to other clusters, but higher credit scores.
- **Cluster 2:** Tend to have moderate branch visit frequency, higher income and average balance, and moderate credit scores.
- Detailed profiles of each cluster provide a basis for targeted marketing or service strategies.

### 3. Relationships Between Financial Behaviors:
- There is a moderate positive correlation between **income** and **average balance** (0.61), indicating that customers with higher income tend to have larger average balances.
- Correlations between other numerical financial behavior variables (transaction count, credit score) and income or average balance are relatively weak.
- Credit score shows weak negative correlations with average balance (-0.19) and transaction count (-0.08).

### 4. Influence of Product Type on Behavior:
- Deeper analysis confirms that **product type** has a significant influence on **branch visit frequency**.
- Customers with **deposit** products (product type 3) have significantly higher average and median branch visit frequencies compared to customers with savings or credit card products. This is likely related to the characteristics of deposit products that may require physical interaction at branches.

### 5. Relationship Between Mobile Banking Usage and Other Behaviors:
- The comparison between mobile banking users and non-users shows **no statistically significant differences** in the means of numerical variables (age, income, average balance, transaction count, branch visit frequency, credit score).
- The distributions of gender and product type are also relatively similar between the two groups.
- This indicates that mobile banking usage in this dataset is not strongly associated with demographic features, product type, or the numerical financial behaviors tested.

### 6. Factors Influencing Credit Score:
- The linear correlation between the numerical features analyzed and credit score is relatively weak.
- However, the analysis of mean feature values per credit score range (box plots) shows **differences in the characteristics of customers with different credit scores**. For example, customers with a credit score of 500 tend to have higher average balances than some other groups, and customers with scores of 700 and 900 tend to have higher branch visit frequencies.
- This relationship is likely non-linear or influenced by feature interactions.

### 7. Identification of High-Value and Risky Customers:
- Based on the 75th percentile criteria for income, balance, and transactions, **1 high-value customer** was identified with specific characteristics (see specific analysis output for details). Due to the very small number, in-depth characteristic analysis is limited to this single customer.
- Based on the 25th percentile criteria for credit score (score <= 600), **44 risky customers** were identified with general characteristics similar to the overall population in terms of average age and income, but with lower credit scores. The distribution of gender, product type, and mobile banking usage in this group is also relatively similar to the general population.

### 8. Potential Cross-selling and Up-selling Opportunities:
- Based on segmentation, opportunities can be targeted: digital services for Cluster 0, branch-based services for Cluster 1, and premium/investment products for Cluster 2.
- Based on product analysis, deposit customers (frequent branch visitors) could be targets for related products or digital solutions for deposit management.
- High-value customers (if the number is significant) are targets for priority services or wealth management.
- Risky customers (low credit score) require a focus on improving financial literacy or secured products.

**Final Conclusion:**

This comprehensive data exploration has revealed various insights about customers, including general characteristics, different behavioral segments, relationships between financial behaviors and product influence, identification of high-value and risky customer groups, and potential business opportunities. These findings provide a strong foundation for strategic decision-making and the development of more targeted initiatives.
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
