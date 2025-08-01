from model_generation.big_conv_model import ConvNet
from model_generation.depthwise_sep_model import ConvNet as DWS_ConvNet
from model_generation.mixed_kernel_conv_model import Mixed_ConvNet
import pandas as pd
from torchinfo import summary


data_1 = pd.read_csv('./data/conv_layers_1_5/high_mac_model_info5.csv').dropna()
data_2 = pd.read_csv('./data/conv_layers_6_10/high_mac_model_info6.csv').dropna()
data_3 = pd.read_csv('./data/misc_conv_models/high_mac_model_info_1_10.csv').dropna()
data_4 = pd.read_csv('./data/expanded_high_mac_2/expanded_high_mac_2.csv').dropna()
data_5 = pd.read_csv('./data/expanded_high_mac/expanded_high_mac_models.csv').dropna()
data_6 = pd.read_csv('./data/mixed_kernel_models_L2/mixed_kernel_models_L2.csv').dropna()
data_7 = pd.read_csv('./data/mixed_kernel_models_L5/mixed_kernel_models_L5.csv').dropna()
data_8 = pd.read_csv('./data/dws/depthwise_sep_models.csv').dropna()
data_9 = pd.read_csv('./data/dws_2/depthwise_sep_models.csv').dropna()

max_macs = 5e10
min_macs = 0

data = pd.concat([data_1, data_2, data_3, data_4, data_5])
data = data[data['Total MACs'].between(min_macs, max_macs)]

mixed_data = pd.concat([data_6, data_7])
mixed_data = mixed_data[mixed_data['Total MACs'].between(min_macs, max_macs)]

dws_data = pd.concat([data_8, data_9])
dws_data = dws_data[dws_data['Total MACs'].between(min_macs, max_macs)]


def get_summary(model, input_size, macs):
    model_summary = summary(model, input_size=(1, 3, input_size, input_size), verbose=0)
    total_mac = model_summary.total_mult_adds
    total_params = model_summary.total_params
    
    if total_mac != macs:
        print(f'Model {name} has an unequal mac count: {str(macs)} vs {str(total_mac)}')
        
    kernels = [0 for _ in range(num_kernels)]
    kernel_macs_cur = [0 for _ in range(num_kernels)]
    kernel_params_cur = [0 for _ in range(num_kernels)]
    cur_conv_params = 0
    cur_linear_params = 0
    cur_linear_macs = 0
    for layer in model_summary.summary_list:
        if layer.class_name == 'Conv2d':
            k = layer.kernel_size[0]
            kernels[k - 1] += 1
            kernel_macs_cur[k - 1] += layer.macs
            kernel_params_cur[k - 1] += layer.num_params
            cur_conv_params += layer.num_params
        if layer.class_name == 'Linear':
            cur_linear_params += layer.num_params
            cur_linear_macs += layer.macs
            
    for i in range(num_kernels):
        kernel_count[i].append(kernels[i])
        kernel_macs[i].append(kernel_macs_cur[i])
        kernels_macs_percentage[i].append(kernel_macs_cur[i] / total_mac * 100)
        kernel_params[i].append(kernel_params_cur[i])
        kernels_params_percentage[i].append(kernel_params_cur[i] / total_params * 100)
    conv_params.append(cur_conv_params)   
    linear_macs.append(cur_linear_macs)
    linear_params.append(cur_linear_params)

num_kernels = 7
kernel_count = [[] for _ in range(num_kernels)]
kernel_macs = [[] for _ in range(num_kernels)]
kernels_macs_percentage = [[] for _ in range(num_kernels)]
kernel_params = [[] for _ in range(num_kernels)]
kernels_params_percentage = [[] for _ in range(num_kernels)]
conv_params = []
linear_params = []
linear_macs = []

if __name__ == "__main__":
    for _, row in data.iterrows():
        name = row['Model Name']
        layers = row['Layers']
        k_size = int(name.split('_K')[1].split('x')[0])
        input_size = int(name.split('W')[1])
        macs = row['Total MACs']
        channels = int(name.split('_F')[1].split('_')[0])
        
        model = ConvNet(layers, channels, (k_size, k_size), input_size)
        get_summary(model, input_size, macs)
        
    for _, row in mixed_data.iterrows():
        name = row['Model Name']
        layers = row['Layers']
        split = name.split('_K')
        channels = int(name.split('_F')[1].split('_')[0])
        input_size = int(name.split('W')[1])
        macs = row['Total MACs']
        k1 = int(split[1].split('x')[0])
        k2 = int(split[2].split('x')[0]) 
        
        model = Mixed_ConvNet(layers, channels, (k1, k1), (k2, k2), input_size)
        get_summary(model, input_size, macs)

    for _, row in dws_data.iterrows():
        name = row['Model Name']
        layers = row['Layers']
        k_size = int(name.split('_K')[1].split('x')[0])
        channels = int(name.split('_F')[1].split('_')[0])
        input_size = int(name.split('W')[1].split('_')[0])
        macs = row['Total MACs']
        
        model = DWS_ConvNet(layers, channels, (k_size, k_size), input_size)
        get_summary(model, input_size, macs)
        
    data = pd.concat([data, mixed_data, dws_data])
    data['Linear MACs'] = linear_macs
    data['Linear Parameters'] = linear_params
    data['Convolution Parameters'] = conv_params
    for i, columns, in enumerate(zip(kernel_count, kernel_macs, kernels_macs_percentage, kernel_params, kernels_params_percentage)):
        kernel_count_col, kernel_macs_col, kernels_macs_percentage_col, kernel_params_col, kernels_params_percentage_col = columns
        k = i + 1
        data[f'K{k}x{k} Count'] = kernel_count_col
        data[f'K{k}x{k} MAC Count'] = kernel_macs_col
        data[f'K{k}x{k} MAC Percentage'] = kernels_macs_percentage_col
        data[f'K{k}x{k} Parameter Count'] = kernel_params_col
        data[f'K{k}x{k} Parameter Percentage'] = kernels_params_percentage_col
    data.to_csv('everything2.csv', index=False)