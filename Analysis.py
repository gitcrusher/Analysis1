import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === Step 1: Data Loading and Preprocessing ===
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Specify the file path
file_path = r'/Dataset.xlsx'

# Verify if the file exists
if os.path.exists(file_path):
    print("File Found!")
    data = pd.read_excel(file_path)
else:
    raise FileNotFoundError(f"File Not Found: {file_path}. Please check the file path and try again.")

# === Step 2: Exploratory Data Analysis (EDA) ===
print("\n===== Exploratory Data Analysis =====")
print("\nDataset Dimensions:", data.shape)
print("\nColumn Names:", list(data.columns))

# Check for duplicate rows
print("\nNumber of Duplicate Rows:", data.duplicated().sum())

# Data types and missing values
print("\nData Types and Missing Values:")
print(data.info(show_counts=True))

# Summary statistics for numerical columns
print("\nNumerical Columns Statistics:")
print(data.describe().round(2))

# === Step 3: Categorical Data Analysis ===
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns Summary:")
for col in categorical_cols:
    print(f"\n{col} - Unique Values:", data[col].nunique())
    print(data[col].value_counts().head())

# Visualize categorical distributions
for col in categorical_cols:
    if data[col].nunique() < 20:  # Avoid plotting for high-cardinality columns
        plt.figure(figsize=(8, 4))
        sns.countplot(data=data, x=col, palette='Set2')
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xticks(rotation=45)
        plt.show()

# === Step 4: Outlier Detection ===
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("\nOutlier Analysis:")
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = len(data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))])
    print(f"{col} - Number of outliers: {outliers}")

# Visualize outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numerical_cols])
plt.title('Outlier Detection', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# === Step 5: Sales and Revenue Analysis ===
print("\nCorrelation Matrix:")
corr_matrix = data[['Gross_sales', 'Net_quantity']].corr()
print(corr_matrix.round(3))

# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.3f')
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

# Scatter plot for Gross_sales vs Net_quantity
plt.figure(figsize=(10, 6))
sns.regplot(x='Net_quantity', y='Gross_sales', data=data, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title('Sales Volume vs Revenue', fontsize=14)
plt.xlabel('Quantity Sold', fontsize=12)
plt.ylabel('Gross Sales (INR)', fontsize=12)
plt.show()

# === Step 6: Key Visualizations ===

# Bar Chart: Sales by Category
category_sales = data.groupby('Category')['Gross_sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
bars = category_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Sales (INR)', fontsize=12)
plt.xticks(rotation=45)
for i, v in enumerate(category_sales):
    plt.text(i, v, f'{v/1e6:.1f}M', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Pie Chart: Payment Mode Distribution
payment_mode = data['Payment_Mode'].value_counts()
payment_mode_top = payment_mode[payment_mode > payment_mode.sum() * 0.05]
payment_mode_top['Others'] = payment_mode[payment_mode <= payment_mode.sum() * 0.05].sum()
plt.figure(figsize=(8, 8))
payment_mode_top.plot(kind='pie', autopct='%1.1f%%')
plt.title('Payment Mode Distribution', fontsize=14)
plt.show()

# Funnel Chart: Sales Funnel Analysis
stages = ['Total Orders', 'Successful Deliveries', 'Premium Shipping']
values = [
    len(data),
    len(data[data['Returns'] == 0]),
    len(data[data['ship_service_level'] == 'Premium'])
]

plt.figure(figsize=(10, 8))
plt.barh(stages, values, color=['#2ecc71', '#3498db', '#9b59b6'])
plt.title('Sales Funnel Analysis', fontsize=14)
plt.xlabel('Number of Orders', fontsize=12)
plt.gca().invert_yaxis()
for i, v in enumerate(values):
    plt.text(v, i, f' {v:,}', va='center', fontsize=10)
plt.tight_layout()
plt.show()

# Funnel Conversion Rates
print("\nFunnel Conversion Rates:")
for i in range(len(stages)-1):
    conversion = (values[i+1] / values[i]) * 100
    print(f"{stages[i]} → {stages[i+1]}: {conversion:.1f}%")

# Pairplot: Multi-variable Analysis
sns.pairplot(data[['Gross_sales', 'Net_quantity', 'Returns']], diag_kind='kde')
plt.suptitle('Multi-variable Analysis', y=1.02, fontsize=16)
plt.show()

# === Step 7: Time Series Analysis ===
# Ensure Date column is parsed correctly
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    yearly_sales = data.groupby('Year')['Gross_sales'].sum().sort_values(ascending=False)
    monthly_sales = data.groupby('Month')['Gross_sales'].mean().sort_index()

    # Yearly Sales Trend
    plt.figure(figsize=(12, 6))
    yearly_sales.plot(kind='bar', color='orange')
    plt.title('Yearly Sales Trend', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Sales (INR)', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Monthly Sales Trend
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(marker='o', color='blue')
    plt.title('Monthly Sales Trend', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Sales (INR)', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.show()

# === Step 8: Geographic Distribution ===
indian_states = ['Maharashtra', 'Tamil Nadu', 'Uttar Pradesh', 'Karnataka', 'Gujarat']
data['State'] = np.random.choice(indian_states, len(data))  # Replace with actual state data if available
state_sales = data.groupby('State')['Gross_sales'].sum().sort_values(ascending=False)

# State-wise Sales
plt.figure(figsize=(12, 6))
bars = state_sales.plot(kind='bar', color='lightgreen')
plt.title('Sales by State', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Total Sales (INR)', fontsize=12)
plt.xticks(rotation=45)
for i, v in enumerate(state_sales):
    plt.text(i, v, f'{v/1e6:.1f}M', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()