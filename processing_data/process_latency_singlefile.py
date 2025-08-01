import pandas as pd
import os
import sys

# Paths to the files
big_csv_path = 'real_models.csv'
power_csv_path = 'tmp/big_power.csv'
latency_csv_path = 'tmp/big_latency.csv'

# Load the big CSV file
big_df = pd.read_csv(big_csv_path)

inferences_dict = {}
latency_dict = {}

# Process power csv file
power_df = pd.read_csv(power_csv_path)
latency_df = pd.read_csv(latency_csv_path)

if power_df.shape[0] != latency_df.shape[0]:
    raise Exception('Different number of power and latency data entries')

for i, power_row, in power_df.iterrows():
    model_name = power_row['model_name']
    latency = latency_df.loc[latency_df['Model Name'] == model_name, 'Average Inference Time (seconds)']
    if latency.shape[0] != 1:
        raise Exception('Missing or dupliate models')
    latency = latency.iloc[0]
    
    inferences_dict[model_name] = power_row['num_inferences']
    latency_dict[model_name] = latency * 1000  # seconds to ms

big_df['num_inferences'] = big_df['Model Name'].map(inferences_dict)
big_df['average_latency(ms)'] = big_df['Model Name'].map(latency_dict)

gpu_power_dict = {}
cpu_power_dict = {}

for i, row, in big_df.iterrows():
    model_name = row['Model Name']
    gpu_power_dict[model_name] = row['Total_POM_5V_GPU'] / row['num_inferences']
    cpu_power_dict[model_name] = row['Total_POM_5V_CPU'] / row['num_inferences']

big_df['GPU_energy_per_inference'] = big_df['Model Name'].map(gpu_power_dict)
big_df['CPU_energy_per_inference'] = big_df['Model Name'].map(cpu_power_dict)

big_df.to_csv(big_csv_path, index=False)