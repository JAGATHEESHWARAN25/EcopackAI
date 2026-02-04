import pandas as pd
import numpy as np

mdf = pd.read_csv('data/materials1.csv')
pdf = pd.read_csv('data/products1.csv')
pdf_2 = pd.read_csv('data/products2.csv')
mdf.info()
pdf.info()
pdf_2.info()
mdf.head()
pdf.head()
pdf_2.head()
pdf_2.isnull().sum()
mdf.isnull().sum()
pdf.isnull().sum()
mdf.describe()
pdf.describe()
pdf_2.describe()
print(pdf['product_category_name'].value_counts())
pdf_clean = pdf.copy()
num_cols = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]
pdf_clean[num_cols] = pdf_clean[num_cols].fillna(
    pdf_clean[num_cols].median()
)
pdf_clean["product_category_name"] = (
    pdf_clean["product_category_name"]
    .fillna(pdf_clean["product_category_name"].mode()[0])
)
pdf_clean.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=pdf_clean[col], color="skyblue")
    plt.title(f"Outliers in {col}")

plt.tight_layout()
plt.show()


def cap_outliers_iqr(df, columns):
    df_capped = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_capped[col] = np.where(
            df[col] < lower, lower,
            np.where(df[col] > upper, upper, df[col])
        )

    return df_capped

def cap_outliers_iqr(df, columns):
    df_capped = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_capped[col] = df[col].clip(lower, upper)

    return df_capped


pdf_outlier_handled = cap_outliers_iqr(pdf_clean, num_cols)

plt.figure(figsize=(14, 10))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=pdf_outlier_handled[col], color="lightgreen")
    plt.title(f"{col} after outlier handling")

plt.tight_layout()
plt.show()