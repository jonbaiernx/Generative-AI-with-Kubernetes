from ydata_quality import DataQuality
from ydata_quality.missings import MissingsProfiler
from ydata_quality.erroneous_data import ErroneousDataIdentifier
from colorama import Fore, Style
import pandas as pd

# Load demo data
df = pd.read_csv('..\demo-data.csv', sep=r'\s*,\s*', header=0, encoding='ascii')

# Set Bias and Erroneous search items
ED_EXTENSIONS = ['Network Eng', 'None', '']
SENSITIVE_FEATURES = ['education']
dq = DataQuality(df=df, label='job_title', ed_extensions=ED_EXTENSIONS, sensitive_features=SENSITIVE_FEATURES, random_state=13)

# Overall quality results
print(Fore.GREEN + "---- Data Quality Summary ----" + Style.RESET_ALL)
results = dq.evaluate()
print("\n")

# Detailed quality issues
print(Fore.GREEN + "---- Label Warning Details ----" + Style.RESET_ALL)
label_warnings = dq.get_warnings(category='Labels')
for lw in label_warnings:
    print(lw.description)
    print("Data:\n")
    print(lw.data)

print(Fore.GREEN + "---- Duplicate Warning Details ----" + Style.RESET_ALL)
duplicate_warnings = dq.get_warnings(category='Duplicates')
for dw in duplicate_warnings:
    print(dw.description)
    print("Data:\n")
    print(dw.data)

print(Fore.GREEN + "---- Erroneous Data Warning Details ----" + Style.RESET_ALL)
erroneous_warnings = dq.get_warnings(category='Erroneous Data')
for ew in erroneous_warnings:
    print(ew.description)
    print("Data:\n")
    print(ew.data)

print(Fore.GREEN + "---- Bias & Fairness Warning Details ----" + Style.RESET_ALL)
bf_warnings = dq.get_warnings(category='Bias&Fairness')
for bfw in bf_warnings:
    print(bfw.description)
    print("Data:\n")
    print(bfw.data)

