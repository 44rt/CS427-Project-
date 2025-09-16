# check_attacks.py
import pandas as pd

df = pd.read_csv('data/Encoded.csv')
print("Unique values in 'Attack Type' column:")
print(df['Attack Type'].value_counts())
print(f"\nTotal unique values: {df['Attack Type'].nunique()}")