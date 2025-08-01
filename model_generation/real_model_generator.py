import torch
import torchvision
from torchinfo import summary
import os
import pandas as pd
#import onnx
import pretrainedmodels


W = 224
channels = 3
onnx_dir = 'real_onnx_models'
output_csv_name = 'real_models.csv'

if not os.path.exists(onnx_dir):
    os.makedirs(onnx_dir)

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg19 = torchvision.models.vgg19(pretrained=True)
shufflenet_v2 = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=False)
unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False, scale=0.5)
mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
resnet50 = torchvision.models.resnet50(pretrained=True)
efficientnet_v1_b4 = torchvision.models.efficientnet_b4(pretrained=True)
efficientnet_v2_m = torchvision.models.efficientnet_v2_m(pretrained=True)

models = {
        'VGG16': vgg16,
        'VGG19': vgg19,
        'Shufflenet_v2': shufflenet_v2,
        'UNet': unet,
        'ResNet50': resnet50,
        "MobileNetV2": mobilenet_v2,
        'Xception': pretrainedmodels.xception(),
        'EfficientNet_v1': efficientnet_v1_b4,
        'EfficientNet_v2': efficientnet_v2_m
    }


model_info = []
for name, model in models.items():
    dummy_input = torch.randn(1, channels, W, W)  # Assuming batch size of 1, 3 channels (RGB)
    onnx_file = os.path.join(onnx_dir, f"{name}.onnx")
    torch.onnx.export(model, dummy_input, onnx_file, verbose=False)
    
    model_summary = summary(model, input_size=(1, 3, W, W), verbose=1)
    
    L = 0
    kernels = [0 for _ in range(7)]
    kernel_macs = [0 for _ in range(7)]
    kernel_params = [0 for _ in range(7)]
    conv_params = 0
    linear_params = 0
    linear_macs = 0
    for layer in model_summary.summary_list:
        if layer.class_name == 'Conv2d':
            L += 1
            k = layer.kernel_size[0]
            kernels[k - 1] += 1
            kernel_macs[k - 1] += layer.macs
            kernel_params[k - 1] += layer.num_params
            conv_params += layer.num_params
        if layer.class_name == 'Linear':
            linear_params += layer.num_params
            linear_macs += layer.macs
    
    total_params = model_summary.total_params
    total_mac = model_summary.total_mult_adds
    
    kernel_macs_percent = [kernel/total_mac * 100 for kernel in kernel_macs]
    kernel_params_percent = [kernel/total_params * 100 for kernel in kernel_params]

    network_type = "ConvNet"
 
    datarow = {
        "Model Name": name,
        "Layers": L,
        "Total Parameters": total_params,
        "Convolution Parameters": conv_params,
        "Total MACs": total_mac,
        "Network Type": network_type,
        "Input Size": W,
        "Linear MACs": linear_macs,
        "Linear Parameters": linear_params
    }
    
    for i in range(7):
        k = i + 1
        datarow[f'K{k}x{k} Count'] = kernels[i]
        datarow[f'K{k}x{k} Count'] = kernels[i]
        datarow[f'K{k}x{k} MAC Count'] = kernel_macs[i]
        datarow[f'K{k}x{k} MAC Percentage'] = kernel_macs_percent[i]
        datarow[f'K{k}x{k} Parameter Count'] = kernel_params[i]
        datarow[f'K{k}x{k} Parameter Percentage'] = kernel_params_percent[i]
 
    model_info.append(datarow)
    df = pd.DataFrame(model_info)
    
df.to_csv(output_csv_name, index=False)