import pandas as pd
import numpy as np
import scipy.stats as st

# Read CSV file
df = pd.read_csv("Book12.csv")
print("Data loaded successfully")

# Compute conversion rates
df['Conversion_Accepted'] = df['Leads accepted'] / df['Leads Created'] * 100
df['Conversion_Installation'] = df['Installed'] / df['Leads accepted'] * 100

# Group by City for average conversion rates
conversion_summary = df.groupby('Lead City').agg({
    'Conversion_Accepted': 'mean',
    'Conversion_Installation': 'mean'
})

# Save the summary and detailed analysis to Excel
with pd.ExcelWriter("output_report.xlsx", engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Data')
    conversion_summary.to_excel(writer, sheet_name='ConversionSummary')
    
print("Excel report generated as output_report.xlsx")