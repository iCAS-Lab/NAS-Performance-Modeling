import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'axes.labelsize': 40})
plt.rcParams.update({'xtick.labelsize': 30})
plt.rcParams.update({'ytick.labelsize': 30})
plt.rcParams.update({'legend.fontsize': 34})


data = pd.read_csv('data/dataset.csv')

# cannot plot data that has multiple kernel sizes (DWS has 1x1 conv)
data = data[~data['Model Name'].str.contains('DWS')]
data = data[data['Model Name'].str.count('_K') == 1]

kernels = ('1x1', '2x2', '3x3', '4x4')
kernel_data = []
for kernel in kernels:
  cur_data = data[data['Model Name'].str.contains(kernel)]
  print(cur_data.shape[0])
  x = cur_data[['Layers', 'Total Parameters', 'Total MACs', 'Input Size', 'Avrg_RAM',]]
  y = cur_data[['GPU_energy_per_inference', 'average_latency(ms)']]
  kernel_data.append((x, y))


figsize = figsize=(14, 11.5)
plt.figure(figsize=figsize)

colors = ['#103778', '#0593A2', '#FF7A48', '#E3371E']
for kernel, data, color in zip(kernels, kernel_data, colors):
  x = data[0]
  y = data[1]
  plt.scatter(x['Total Parameters'], y['GPU_energy_per_inference'], label=kernel, c=color)
  
plt.xlabel('Total Parameters')
plt.ylabel('GPU Energy per Inference (mJ)')
# plt.title('Scatter plot between Total Parameters and GPU Power')
plt.legend()

plt.figure(figsize=figsize)
for kernel, data, color in zip(kernels, kernel_data, colors):
  x = data[0]
  y = data[1]
  plt.scatter(x['Total Parameters'], y['average_latency(ms)'], label=kernel, c=color)

plt.xlabel('Total Parameters')
plt.ylabel('Inference Latency (ms)')
# plt.title('Scatter plot between Total Parameters and Average Latency')
plt.legend()

plt.figure(figsize=figsize)

colors = ['#103778', '#0593A2', '#FF7A48', '#E3371E']
for kernel, data, color in zip(kernels, kernel_data, colors):
  x = data[0]
  y = data[1]
  plt.scatter(x['Total MACs'], y['GPU_energy_per_inference'], label=kernel, c=color)
  
plt.xlabel('Number of MAC Operations')
plt.ylabel('GPU Energy per Inference (mJ)')
# plt.title('Scatter plot between Total MACs and GPU Power')
plt.legend()

plt.figure(figsize=figsize)
for kernel, data, color in zip(kernels, kernel_data, colors):
  x = data[0]
  y = data[1]
  plt.scatter(x['Total MACs'], y['average_latency(ms)'], label=kernel, c=color)

plt.xlabel('Number of MAC Operations')
plt.ylabel('Inference Latency (ms)')
# plt.title('Scatter plot between Total MACs and Average Latency')
plt.legend()

plt.show()
