import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
import sys
import csv

if len(sys.argv) < 2:
    print("Directory not specified")
    print("Usage:", sys.argv[0], "<directory_with_trt_models>")
    exit()

directory = sys.argv[1]

# Create the CSV filename by appending 'latency' to the folder name
folder_name = os.path.basename(os.path.normpath(directory))
results_file = f"{folder_name}_latency.csv"

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        
# Prepare the CSV file to store results
with open(results_file, mode='w', newline='') as csv_file:
    fieldnames = ['Model Name', 'Average Inference Time (seconds)', 'Model Load Time (seconds)', 'Allocation Time (seconds)']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    for filename in os.listdir(directory):
        if filename.endswith('.trt'):
            model_name = filename[:-4]  # Remove the '.trt' extension

            # Load the TensorRT engine
            with open(os.path.join(directory, filename), 'rb') as f:
                engine_bytes = f.read()
                engine = runtime.deserialize_cuda_engine(engine_bytes)

            # Create execution context
            context = engine.create_execution_context()

            # Get input shape and allocate input tensor
            input_shape = engine.get_binding_shape(0)  # Assuming the first binding is the input
            input_size = np.prod(input_shape)  # Total number of elements
            input_dtype = trt.nptype(engine.get_binding_dtype(0))
            inp_rand = np.random.rand(*input_shape).astype(input_dtype)

            currentBaseTime = time.time()
            interpreterReadTime = (time.time() - currentBaseTime) * 1000  # in milliseconds

            inputs = []
            outputs = []
            bindings = []
            currentBaseTime = time.time()
            stream = cuda.Stream()

            # Allocate host and device buffers
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(1 * size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                # Append the device buffer to device bindings.
                bindings.append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    inputs.append(HostDeviceMem(host_mem, device_mem))
                else:
                    outputs.append(HostDeviceMem(host_mem, device_mem))

            allocationTime = (time.time() - currentBaseTime) * 1000  # in milliseconds

            def infer(input_data):
                np.copyto(inputs[0].host, input_data.flatten())
                [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
                context.execute_async(bindings=bindings, stream_handle=stream.handle)
                [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
                stream.synchronize()
                return [out.host for out in outputs]

            num_runs = 100
            total_inference_time = 0

            # do not count first inference
            _ = infer(inp_rand)

            for _ in range(num_runs):
                start_time = time.time()
                _ = infer(inp_rand)
                total_inference_time += (time.time() - start_time)

            avg_inference_time = total_inference_time / num_runs  # in seconds

            # Write the results to the CSV file
            writer.writerow({
                'Model Name': model_name,
                'Average Inference Time (seconds)': avg_inference_time,
                'Model Load Time (seconds)': interpreterReadTime / 1000,
                'Allocation Time (seconds)': allocationTime / 1000
            })

print(f"Inference results have been saved to {results_file}.")
