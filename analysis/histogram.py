import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer


data = pd.read_csv('../analysis/dataset.csv')
print(data.shape[0])

mac_operations = data['Total MACs']

plt.figure(figsize=(8, 6))
plt.hist(mac_operations, bins=100, edgecolor='black', log=True)  # log=True for Y-axis log scale
plt.xlabel('Number of MAC Operations')
plt.ylabel('Frequency')
plt.title('Distribution of MAC Operations')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
