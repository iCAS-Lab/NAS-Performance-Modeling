import torch
import torch.nn as nn
import torch.onnx
import pandas as pd
from torchinfo import summary
import itertools
import os
 
# Function to calculate tensor size in bytes
 
def calculate_tensor_size(batch_size, channels, width, height, data_type_size=4):
    return batch_size * channels * width * height * data_type_size
 
class DepthwiseSeperableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
 
class ConvNet(nn.Module):
    def __init__(self, L, F, K, input_size):
        super(ConvNet, self).__init__()
        layers = []
        in_channels = 3  # Assuming RGB input with 3 channels
        padding=2
        current_size = input_size
        for _ in range(L):
            if K == (3,3):
                padding=1
            layers.append(DepthwiseSeperableConv2d(in_channels, F, kernel_size=K, padding=padding))
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
    onnx_dir = 'new_onnx_models6'
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)

    L_layers = range(5,11)  # Number of layers (1 to 5)
    F_filters = [256, 288, 320, 352, 384, 416, 448, 480, 512]
    K_kernel_sizes = [(2,2), (3,3), (4,4), (5,5)]
    W_input_sizes = range(10, 101, 5)  
    
    max_tensor_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    min_macs = 1e9
    max_macs = 5e10
    model_info = []
    onnx_files_to_delete = []
    for i, (L, F, K, W) in enumerate(itertools.product(L_layers, F_filters, K_kernel_sizes, W_input_sizes), 1):
        tensor_size = calculate_tensor_size(batch_size=1, channels=3, width=W, height=W)
        if tensor_size > max_tensor_size_bytes:
            print(f"Skipping model convnet_L{L}_F{F}_K{K[0]}x{K[1]}_W{W} due to tensor size exceeding 2GB.")
            continue
        
        model_name = f"convnet_L{L}_F{F}_K{K[0]}x{K[1]}_W{W}_DWS"
        
        model = ConvNet(L, F, K, W)
        
        model_summary = summary(model, input_size=(1, 3, W, W), verbose=0)
        total_params = model_summary.total_params
        total_mac = model_summary.total_mult_adds
        conv_params = 0
        for layer in model_summary.summary_list:
            if layer.class_name == 'Conv2d':
                conv_params += layer.num_params
    
        if total_mac < min_macs or total_mac > max_macs:
            print(f"Skipping model {model_name} due to insufficient MACs out of the range: {total_mac}.")
            continue
        
        dummy_input = torch.randn(1, 3, W, W)  # Assuming batch size of 1, 3 channels (RGB)
        onnx_file = os.path.join(onnx_dir, f"{model_name}.onnx")
        try:
            torch.onnx.export(model, dummy_input, onnx_file, verbose=False)
        except:
            continue
    
        network_type = "ConvNet"
    
        model_info.append({
            "Model Name": model_name,
            "Layers": L,
            "Total Parameters": total_params,
            "Convolution Parameters": conv_params,
            "Total MACs": total_mac,
            "Network Type": network_type,
            "Input Size": W
        })
        df = pd.DataFrame(model_info)
        df.to_csv("depthwise_sep_models.csv", index=False)
        
    print(model_info)
    print("New model generation complete. ONNX files and CSV saved for models with MACs in defined range and tensor size < 2GB.")