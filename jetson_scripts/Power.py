import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os
import sys
import lightweight_dataframes as dataframes
import subprocess

# Check for command-line arguments
if len(sys.argv) < 4:
    print("File not specified or inference time not provided")
    print("Usage:", sys.argv[0], "<model.trt> <inference_time> <power_output_file_name>")
    exit()

# Benchmarking information
print("Benchmarking power for", sys.argv[1])
print("duration: " ,sys.argv[2])

# Initialize variables
filename = sys.argv[1]
inference_time = int(sys.argv[2])
output_file = sys.argv[3]
model_name = filename[:-4]

# Generate input data (example: random data)
#val_x = np.random.rand(100, 1, 3, 224, 224)

# Timing and logging setup
start_time = time.time()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

# Deserialize CUDA engine from file
with open(filename, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)

# Create execution context and allocate memory
context = engine.create_execution_context()
batch_size = 1

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

input_shape = tuple(engine.get_binding_shape(0))
print("Input shape:", input_shape)
val_x = np.random.rand(100, *input_shape)

# Allocate memory for inputs and outputs
inputs = []
outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    # Allocate host and device buffers
    print("Size: ", size)
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    # Append the device buffer to device bindings
    bindings.append(int(device_mem))

    # Append to the appropriate list
    if engine.binding_is_input(binding):
        inputs.append(HostDeviceMem(host_mem, device_mem))
    else:
        outputs.append(HostDeviceMem(host_mem, device_mem))

# Measure model load time
modelLoadTime = (time.time() - start_time)
def infer(input_img):
    input_img = input_img.flatten()
    np.copyto(inputs[0].host, input_img)

    # Asynchronous memory copies and execution
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize stream to wait for inference to complete
    stream.synchronize()

    # Return host outputs
    return [out.host for out in outputs]

# Timing and logging for inference
inference_time = int(sys.argv[2])
df = dataframes.createDataFrame(columnNames=["model_name", "file_name", "num_inferences", "total_time (s)", "load_time (s)", "alloc_time (s)", "inference_time (s)", "average_latency (s)"])
total_inference_latency = 0
num_infer = 0
avg_time = 0

# Timing allocation
currentBaseTime = time.time()
allocationTime = (time.time() - currentBaseTime)

# do not count first inference
inp_id = np.random.randint(100)
input_img = val_x[inp_id]
outs = infer(input_img)

# start collecting power data

log = open(output_file, 'a')
tegrastats = subprocess.Popen(f"sudo tegrastats > {output_file} &", shell=True)
tegrastats_pid = tegrastats.pid

# Start inference timing
inference_start_time = time.time()

# Perform inference for specified time
while time.time() - inference_start_time < inference_time:
    inp_id = np.random.randint(100)
    input_img = val_x[inp_id]
    single_inference_start = time.time()
    outs = infer(input_img)
    single_inference_latency = time.time() - single_inference_start
    total_inference_latency += single_inference_latency
    num_infer += 1

subprocess.run('sudo pkill tegrastats', shell=True, check=True)

# Measure total inference time and program runtime
inferenceTime = time.time() - inference_start_time
totalTime = time.time() - start_time
average_latency = total_inference_latency / num_infer

# Print results and log to DataFrame
print(filename)
print("-----------------")
print("POWER PROFILE")
print("")
print("Number of samples analyzed:\t", num_infer)
print("Model Load Time:\t", modelLoadTime, "(s)")
print("Model Alloc Time:\t", allocationTime, "(s)")
print("Model Inference Time:\t", inferenceTime, "(s)")
print("Average Inference Latency:\t", average_latency, "(s)")
print("Total Program Runtime:\t", totalTime, "(s)")

# Append results to DataFrame and save as CSV
df = dataframes.append_row(df, {"model_name": model_name, "file_name": filename, "num_inferences": num_infer, "total_time (s)": totalTime, "load_time (s)": modelLoadTime, "alloc_time (s)": allocationTime, "inference_time (s)": inferenceTime, "average_latency (s)": average_latency})
dataframes.to_csv(df, model_name + str(inference_time) +  "_power_timeline.csv")



