import torch
import torch.onnx
import torchvision
import joblib
from torch import nn
from torchinfo import summary
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count
    
class fitnessFunction:
    @staticmethod   
    def input_for_predictive_model(model, choice, kchoice, kernel_choice, input_size=(3, 32, 32), device='cuda:0'):
        dummy_input = torch.randn(1, *input_size).to(device)
        output, subnet = model(dummy_input, choice, kchoice, kernel_choice)
        subnet = subnet.to(device)
        model_summary = summary(subnet, input_data=(dummy_input), verbose=0)
        num_kernels = 7

        total_mac = model_summary.total_mult_adds
        total_params = model_summary.total_params
        result = {
            'Total Parameters': total_params,
            'Total MACs': total_mac,
        }

        kernel_macs_cur = [0 for _ in range(num_kernels)]
        for layer in model_summary.summary_list:
            if layer.class_name == 'Conv2d':
                k = layer.kernel_size[0]
                kernel_macs_cur[k - 1] += layer.macs
        for i in range(num_kernels):
            k = i + 1
            mac_pct = (kernel_macs_cur[i] / total_mac * 100) if total_mac > 0 else 0.0
            result[f'K{k}x{k} MAC Percentage'] = mac_pct

        print("Result:", result)
        return result, subnet       
    
    @staticmethod
    def predict_latency_power(features):
        rf_model = joblib.load("models/best_single_model.joblib")
        input_vector = [features[k] for k in [
            'Total Parameters', 'Total MACs',
            'K1x1 MAC Percentage', 'K2x2 MAC Percentage', 'K3x3 MAC Percentage',
            'K4x4 MAC Percentage', 'K5x5 MAC Percentage', 'K6x6 MAC Percentage', 'K7x7 MAC Percentage'
        ]]
        prediction = rf_model.predict([input_vector])[0]
        # print("prediction: ", prediction)
        power, latency = prediction[0], prediction[1]
        return latency, power

    @staticmethod
    def measure_latency_power_on_jetson(subnet, dummy_input, onnx_path='subnet.onnx', trt_script='run_trt_and_measure.sh'):
        torch.onnx.export(subnet, dummy_input, onnx_path, opset_version=11)
        # Assumes SSH access and script on Jetson Nano handles inference and returns latency/power
        output = subprocess.check_output(["bash", trt_script, onnx_path])
        output = output.decode('utf-8').strip()
        latency, power = map(float, output.split(','))
        return latency, power
