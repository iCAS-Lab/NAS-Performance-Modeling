import tensorrt as trt
import sys

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Parse the ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Parsed the onnx model successfully!")
    
    """
    has_dynamic_shape = False
    for i in range(network.num_inputs):
        shape = network.get_input(i).shape
        if any(dim == -1 for dim in shape):
            has_dynamic_shape = True
            break

    if has_dynamic_shape:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            shape = input_tensor.shape
            min_shape = [s if s != -1 else 1 for s in shape]
            print(min_shape)
            opt_shape = [s if s != -1 else 4 for s in shape]
            print(opt_shape)
            max_shape = [s if s != -1 else 8 for s in shape]
            print(max_shape)
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    """


    # Build the TensorRT engine
    print('Building TensorRT engine. This may take a few minutes...')
    engine = builder.build_engine(network, config)
    if engine is None:
        print('Failed to build the engine.')
        return None

    # Save the engine to file
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    print('TensorRT engine built and saved at', engine_file_path)
    return engine

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path_to_onnx_model> <path_to_save_trt_engine>")
        sys.exit(1)

    onnx_file_path = sys.argv[1]
    engine_file_path = sys.argv[2]

    build_engine(onnx_file_path, engine_file_path)
