from collections import Counter
import pandas as pd


df = pd.read_csv("PJI_Dataset/New_data_y_test.csv")
count_class = Counter(df["Group"])
print(count_class)
