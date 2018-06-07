# loading necessary libararies
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from Map_Variable_by_Stage import d_dvd, d_va

from datacleaning import merged_dataset, dvd_original, va_original

"""Run some preliminary diagnostics on the two datasets."""

test_dvd = dvd_original[['demo_ID','demo_age', 'Trans_num', 'Cancer_Gleason', 'Cancer_psa']]
test_va = va_original[['ID','age', 'AvailTransc', 'gleason', 'psa1']]

test_merged = merged_dataset[['ID','age', 'AvailTransc', 'gleason', 'psa1']]


print("DVD")
print(pd.DataFrame.describe(test_dvd))
print()
print("VA")
print(pd.DataFrame.describe(test_va))
print()
print("Merged")
print(pd.DataFrame.describe(test_merged))


#print(len(dvd_original['Trans_num'] == 2))
print()
print("Number of DVD patients with transcript data:",(dvd_original.Trans_num == 1).sum() + (dvd_original.Trans_num == 2).sum())
print("Number of VA patients with transcript data:", (va_original.AvailTransc == 1).sum())

print("Number of Merged patients with transcript data:", (merged_dataset.AvailTransc == 1).sum() + (merged_dataset.AvailTransc == 2).sum())

#print("Number of total patients without transcript data:", (merged_dataset.AvailTransc == 0).sum())


#print(pd.DataFrame.describe(merged_dataset))

