import re
import os
import pandas as pd
import numpy as np


def parse_tegrastats_line(line, total_stats):
    stats = {}
    
    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    if ram_match:
        stats['RAM'] = int(ram_match.group(1))

    
    swap_match = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if swap_match:
        stats['SWAP'] = int(swap_match.group(1))

    
    iram_match = re.search(r'IRAM (\d+)/(\d+)kB', line)
    if iram_match:
        stats['IRAM'] = int(iram_match.group(1))

    
    cpu_match = re.findall(r'CPU \[(.*?)\]', line)
    if cpu_match:
        cpu_usages = re.findall(r'(\d+)%@\d+', cpu_match[0])
        stats['CPU'] = np.mean([int(usage) for usage in cpu_usages])

    
    gr3d_freq_match = re.search(r'GR3D_FREQ (\d+)%@', line)
    if gr3d_freq_match:
        stats['GPU'] = int(gr3d_freq_match.group(1))

    
    emc_freq_match = re.search(r'EMC_FREQ (\d+)%@', line)
    if emc_freq_match:
        stats['EMC_FREQ'] = int(emc_freq_match.group(1))

    
    ape_match = re.search(r'APE (\d+)', line)
    if ape_match:
        stats['APE'] = int(ape_match.group(1))

    
    temps = {
        'PLL': r'PLL@(\d+)C',
        'CPU_TEMP': r'CPU@(\d+)C',
        'PMIC': r'PMIC@(\d+)C',
        'GPU_TEMP': r'GPU@(\d+)C',
        'AO': r'AO@(\d+\.\d+)C',
        'THERMAL': r'thermal@(\d+\.\d+)C'
    }
    for key, regex in temps.items():
        match = re.search(regex, line)
        if match:
            stats[key] = float(match.group(1))

    
    pom_in_match = re.search(r'POM_5V_IN (\d+)/', line)
    if pom_in_match:
        stats['POM_5V_IN'] = int(pom_in_match.group(1))
        total_stats['Total_POM_5V_IN'] += stats['POM_5V_IN']

    pom_gpu_match = re.search(r'POM_5V_GPU (\d+)/', line)
    if pom_gpu_match:
        stats['POM_5V_GPU'] = int(pom_gpu_match.group(1))
        total_stats['Total_POM_5V_GPU'] += stats['POM_5V_GPU']

    pom_cpu_match = re.search(r'POM_5V_CPU (\d+)/', line)
    if pom_cpu_match:
        stats['POM_5V_CPU'] = int(pom_cpu_match.group(1))
        total_stats['Total_POM_5V_CPU'] += stats['POM_5V_CPU']

    return stats


def process_tegrastats_file(file_path, total_stats):
    stats_list = []
    line_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            stats = parse_tegrastats_line(line, total_stats)
            if stats:
                stats_list.append(stats)
                line_count += 1

    if stats_list and line_count > 0:
        df = pd.DataFrame(stats_list)
        avg_stats = df.mean().to_dict()
        return avg_stats, line_count
    else:
        return None, 0


def update_model_row_in_csv(df, model_txt, model_name):
    total_stats = {}
    total_stats['Total_POM_5V_IN'] = 0
    total_stats['Total_POM_5V_GPU'] = 0
    total_stats['Total_POM_5V_CPU'] = 0

    avg_stats, line_count = process_tegrastats_file(model_txt, total_stats)

    if avg_stats:
        row_index = df[df['Model Name'] == model_name].index
        if len(row_index) == 0:
            print(f"Model {model_name} not found in CSV.")
        else:
            row_index = row_index[0]
            for key, value in avg_stats.items():
                avg_col_name = f"Avrg_{key}"
                if avg_col_name not in df.columns:
                    df[avg_col_name] = np.nan
                
                df.at[row_index, avg_col_name] = value

            print(f"Averages updated for model {model_name} based on {line_count} readings.")
            
            for key, value in total_stats.items():
                if key not in df.columns:
                    df[key] = np.nan
                
                df.at[row_index, key] = value
                
            print(f"Total power updated for model {model_name} based on {line_count} readings.")
    else:
        print(f"No valid data found in {model_txt}")


def process_folder_of_txt_files(folder_path, model_csv):
    df = pd.read_csv(model_csv)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            model_name = file_name[len("stats_"):]
            model_name = model_name.split('_mode')[0]
            #model_name += ".onnx"
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_path} for model {model_name}")
            update_model_row_in_csv(df, file_path, model_name)

    df.to_csv(model_csv, index=False)
    print(f"All model data updated and saved to {model_csv}")


folder_path = 'tmp/'  
model_csv_path = 'real_models.csv'

process_folder_of_txt_files(folder_path, model_csv_path)
