import torch
import torch.nn as nn
import torch.onnx
import pandas as pd
from torchinfo import summary
import itertools
import os
import random
 
# Function to calculate tensor size in bytes
 
def calculate_tensor_size(batch_size, channels, width, height, data_type_size=4):
    return batch_size * channels * width * height * data_type_size
 
class Mixed_ConvNet(nn.Module):
    def __init__(self, L, F, K1, K2, input_size):
        super(Mixed_ConvNet, self).__init__()
        layers = []
        in_channels = 3  # Assuming RGB input with 3 channels
        padding=2
        current_size = input_size
        
        self.count1 = 0
        self.count2 = 0
        
        kernel_choice = True
        for _ in range(L):
            if kernel_choice:
                K = K1
                self.count1 += 1
            else:
                K = K2
                self.count2 += 1
            kernel_choice = not kernel_choice
            
            if K[0] % 2 == 1:  # odd kernel size
                padding=1
                
            layers.append(nn.Conv2d(in_channels, F, kernel_size=K, stride=1, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            in_channels = F
            current_size = (current_size - K[0]+2*padding)//1+1
        self.conv_layers = nn.Sequential(*layers)
        flattened_size = F*current_size*current_size
        self.fc = nn.Linear(flattened_size, 10)
 
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        return x
 
if __name__ == "__main__":
    onnx_dir = 'mixed_kernel_models_dir'
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    
    num_models = 0

    L_layers = range(11,21)  # Number of layers (1 to 5)
    F_filters = [64, 96, 128, 160, 192, 224, 256, 288, 320]
    K_kernel_sizes = [(3,3), (5,5), (7,7)]
    W_input_sizes = range(10, 101, 5)  

    max_tensor_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    min_macs = 1e9
    max_macs = 5e10
    model_info = []

    max_layers = 12
    layer_columns = [f"Layer {i} {feature}" for i in range(1, max_layers) for feature in ["MACs", "Parameters", "Kernel Size", 'Input Size', 'Output Size', 'Class Name']]
    layers_to_view = ['Linear', 'Conv2d', 'BatchNorm2d']

    exhausted_combos = []

    set_kernel = [(1,1)]

    for i, (L, F, K1, K2, W) in enumerate(itertools.product(L_layers, F_filters, set_kernel, K_kernel_sizes, W_input_sizes), 1):
        if num_models >= 4000:
            exit()
        
        model_name_tmp = f"convnet_L{L}_F{F}_K{K1[0]}x{K1[1]}_K{K2[0]}x{K2[1]}_W{W}"
        if K1 == K2:
            print(f"Skipping model {model_name_tmp} due to repeated kernels.")
            continue
        
        tensor_size = calculate_tensor_size(batch_size=1, channels=3, width=W, height=W)
        if tensor_size > max_tensor_size_bytes:
            print(f"Skipping model {model_name_tmp} due to tensor size exceeding 2GB.")
            continue
        
        try:
            model = Mixed_ConvNet(L, F, K1, K2, W)
            model_name = f"convnet_L{L}_F{F}_K{K1[0]}x{K1[1]}_{str(model.count1)}_K{K2[0]}x{K2[1]}_{str(model.count2)}_W{W}"
            
            network_type = "ConvNet"
            model_summary = summary(model, input_size=(1, 3, W, W), verbose=0)
            total_params = model_summary.total_params
            total_mac = model_summary.total_mult_adds
            
            datarow = {
                "Model Name": model_name,
                "Layers": L,
                "Total Parameters": total_params,
                "Total MACs": total_mac,
                "Network Type": network_type,
                "Input Size": W
            }
            
            if total_mac < min_macs or total_mac > max_macs:
                print(f"Skipping model {model_name} due to insufficient MACs out of the range: {total_mac}.")
                del model
                continue
            
            # prevent having ..K1x1_K2x2.. and ..K2x2_K1x1
            # not most effecient way, do not want to deal with updating iteration
            id = set((L, F, K1, K2, W))
            if id in exhausted_combos:
                print(f"Skipping model {model_name_tmp} due repeated flipped configuration")
                del model
                continue
            exhausted_combos.append(id)
            
            dummy_input = torch.randn(1, 3, W, W)  # Assuming batch size of 1, 3 channels (RGB)
            onnx_file = os.path.join(onnx_dir, f"{model_name}.onnx")
            torch.onnx.export(model, dummy_input, onnx_file, verbose=False)
            
        except:
            pass
        
        model_info.append(datarow)
        df = pd.DataFrame(model_info)
        df.dropna().to_csv("mixed_kernel_models_11_20.csv", index=False)
        num_models += 1
        del model
        
    print("New model generation complete. ONNX files and CSV saved for models with MACs in defined range and tensor size < 2GB.")