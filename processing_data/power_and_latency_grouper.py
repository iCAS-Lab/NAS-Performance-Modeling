import os
import sys
import csv

if len(sys.argv) < 3:
    print("Power or latency file not specified")
    print("Usage:", sys.argv[0], "<input_output_directory> <power_file_name> <latency_file_name>")
    exit()

csv_dir = sys.argv[1]
power_fname = sys.argv[2]
latency_fname = sys.argv[3]
power_path = os.path.join(csv_dir, power_fname)
latency_path = os.path.join(csv_dir, latency_fname)

# ---------------- Power ---------------- 
power_files = [file for file in os.listdir(csv_dir) if "power" in file       and \
                                                     not file == power_fname and \
                                                     not file == latency_fname] 
exists = os.path.exists(power_path)

with open(power_path, mode='a', newline='') as power_file:
    fieldnames = ["model_name", 
                  "file_name", 
                  "num_inferences", 
                  "total_time (s)", 
                  "load_time (s)", 
                  "alloc_time (s)", 
                  "inference_time (s)",
                  "average_latency (s)"]
    
    writer = csv.DictWriter(power_file, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    
    for file in power_files:
        file_path = os.path.join(csv_dir, file)
        with open(file_path, mode='r') as small_power_file:
            reader = csv.DictReader(small_power_file, fieldnames=fieldnames)
            for index, row in enumerate(reader):
                if index == 0: # skip header
                    continue
                writer.writerow(row)
                print(row)
        os.remove(file_path)
                
# ---------------- Latency ---------------- 
latency_files = [file for file in os.listdir(csv_dir) if "latency" in file   and \
                                                     not file == power_fname and \
                                                     not file == latency_fname] 
exists = os.path.exists(latency_path)

with open(latency_path, mode='a', newline='') as latency_file:
    fieldnames = ["Model Name",
                  "Average Inference Time (seconds)",
                  "Model Load Time (seconds)",
                  "Allocation Time (seconds)"]
    
    writer = csv.DictWriter(latency_file, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    
    for file in latency_files:
        file_path = os.path.join(csv_dir, file)
        with open(file_path, mode='r') as small_latency_file:
            reader = csv.DictReader(small_latency_file, fieldnames=fieldnames)
            for index, row in enumerate(reader):
                if index == 0: # skip header
                    continue
                writer.writerow(row)
        os.remove(file_path)
