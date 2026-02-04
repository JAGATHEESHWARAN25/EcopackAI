import pandas as pd
import numpy as np

mdf = pd.read_csv('data/materials1.csv')
pdf = pd.read_csv('data/products1.csv')
pdf_2 = pd.read_csv('data/products3.csv')

print(mdf.info())

print(pdf.info())

print(pdf_2.info())

print(mdf.head())

print(pdf.head())

print(pdf_2.head())
