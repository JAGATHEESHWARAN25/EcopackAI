import pandas as pd
import numpy as np
df=pd.read_csv('data/material.csv')
print(df.info())
print(df.head())
print(df.tail())
df.shape
df.duplicated().sum()
df.isnull().sum()