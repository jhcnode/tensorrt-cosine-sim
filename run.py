import numpy as np
import cupy as cp
from sklearn.preprocessing import normalize
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import torch
import torchvision.models as models
import os
import pickle
import glob
import random
import time
from collections import defaultdict

# 글로벌 CUDA 컨텍스트 설정
cuda.init()
device = cuda.Device(0)  # GPU 장치 ID
global_context = device.make_context()

# CuPy가 TensorRT와 동일한 컨텍스트를 사용하도록 설정
cp.cuda.runtime.setDevice(0)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Stratified split dataset from fixed count
def stratified_split_fixed_count(data, train_count, test_count, random_state=42):
    grouped_data = defaultdict(list)
    for class_id, path in data:
        grouped_data[class_id].append(path)

    train_data = []
    test_data = []

    random.seed(random_state)

    for class_id, paths in grouped_data.items():
        random.shuffle(paths)

        train_split_count = min(len(paths), train_count // len(grouped_data))
        test_split_count = min(len(paths) - train_split_count, test_count // len(grouped_data))

        train_data.extend([[class_id, path] for path in paths[:train_split_count]])
        test_data.extend([[class_id, path] for path in paths[train_split_count:train_split_count + test_split_count]])

    if len(train_data) < train_count:
        remaining_train = random.sample(data, train_count - len(train_data))
        train_data.extend(remaining_train)
    if len(test_data) < test_count:
        remaining_test = random.sample(data, test_count - len(test_data))
        test_data.extend(remaining_test)

    train_data.sort(key=lambda x: x[0])
    test_data.sort(key=lambda x: x[0])

    return train_data[:train_count], test_data[:test_count]

# Preprocess images for input into the model
def preprocess_images(image_paths, input_shape):
    transform = transforms.Compose([
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_and_transform(item):
        _, path = item
        image = Image.open(path).convert('RGB')
        return transform(image).numpy()

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_and_transform, image_paths))
    return np.stack(images)

# Builds a TensorRT engine from an ONNX model.
def build_engine(onnx_file_path, max_batch_size=4, enable_fp16=True, enable_int8=False, calibration_data=None, engine_file_path="/kaggle/working/model.engine"):
    if os.path.exists(engine_file_path):
        print(f"Loading engine from file: {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            serialized_engine = f.read()
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            if engine is None:
                print("ERROR: Failed to deserialize the CUDA engine.")
                return None
            print("Success: Loaded the CUDA engine from file.")
            return engine

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if enable_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if enable_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            if calibration_data is not None:
                calibrator = Calibrator(calibration_data, (3, 224, 224), max_batch_size)
                config.int8_calibrator = calibrator

        with open(onnx_file_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(f"Parser Error[{i}]: {parser.get_error(i)}")
                return None

        for i in range(network.num_outputs):
            out_tensor = network.get_output(i)
            network.unmark_output(out_tensor)

        intermediate_layer_name = "/Flatten"
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name.find(intermediate_layer_name)!=-1:
                print("sucess mak")
                network.mark_output(layer.get_output(0))
                break

        if network.num_inputs < 1:
            print("ERROR: The network does not have any input tensors.")
            return None

        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, (1, 3, 224, 224), (max_batch_size, 3, 224, 224), (max_batch_size, 3, 224, 224))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("ERROR: Failed to build the serialized engine.")
            return None

        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
            print(f"Success: Serialized engine saved to {engine_file_path}.")

        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            if engine is None:
                print("ERROR: Failed to deserialize the CUDA engine.")
            else:
                print("Success: Deserialized the CUDA engine.")
            return engine

# Extract features in batch
def extract_features_batch(engine, images, batch_size):
    input_shape = (batch_size, 3, 224, 224)
    input_data = images.astype(np.float32)
    input_host = input_data.ravel()
    d_input = cuda.mem_alloc(input_host.nbytes)
    d_output = cuda.mem_alloc(batch_size * 2048 * np.float32().nbytes)

    cuda.memcpy_htod(d_input, input_host)

    context = engine.create_execution_context()
    context.set_binding_shape(0, input_shape)

    bindings = [int(d_input), int(d_output)]
    context.execute_v2(bindings)

    output_host = np.empty((batch_size, 2048), dtype=np.float32)
    cuda.memcpy_dtoh(output_host, d_output)

    d_input.free()
    d_output.free()

    return output_host

# Calculates cosine similarity for batch processing
def calculate_similarity_batch(network_features, prepared_vectors, use_gpu=True, device_id=0):
    def safe_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    if use_gpu and cp.cuda.runtime.getDeviceCount() > 0:
        cp.cuda.runtime.setDevice(device_id)
        network_features_gpu = cp.array(network_features, dtype=cp.float32)
        prepared_vectors_gpu = cp.array(prepared_vectors, dtype=cp.float32)

        network_features_gpu = safe_normalize(network_features_gpu)
        prepared_vectors_gpu = safe_normalize(prepared_vectors_gpu)

        similarity = cp.sum(network_features_gpu * prepared_vectors_gpu, axis=1).get()
    else:
        network_features = safe_normalize(network_features)
        prepared_vectors = safe_normalize(prepared_vectors)

        similarity = np.sum(network_features * prepared_vectors, axis=1)
    return similarity

# Generates feature vectors from a dataset.
def gen_feat_vec(data,tenssort_path,feat_vect_path,onnx_model_path):
    model = models.resnet50(pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, onnx_model_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13
    )

    batch_size = 200
    engine = None
    save_data = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        calibration_data = None

        if engine is None:
            calibration_data = preprocess_images(batch, (3, 224, 224))
            engine = build_engine(
                onnx_model_path,
                max_batch_size=batch_size,
                enable_fp16=True,
                enable_int8=False,
                calibration_data=calibration_data,
                engine_file_path=tenssort_path
            )
            if engine is None:
                raise RuntimeError("Failed to build the engine.")

        images = preprocess_images(batch, (3, 224, 224)) if calibration_data is None else calibration_data
        network_features = extract_features_batch(engine, images, batch_size=batch_size)
        print(network_features.shape)
        cuda.Context.synchronize()

        for j in range(len(network_features)):
            save_data.append([batch[j][1], batch[j][0], network_features[j]])

    with open(feat_vect_path, 'wb') as file:
        pickle.dump(save_data, file)

# Processes queries by performing feature extraction and similarity matching
def query_data(data, feat_vect_path, tenssort_path,output_path,onnx_model_path):
    
    model = models.resnet50(pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, onnx_model_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    with open(feat_vect_path, 'rb') as file:
        feat_data = pickle.load(file)

    batch_size = 200
    engine = None
    feature_vectors = np.array([item[-1] for item in feat_data])
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        calibration_data = None
        if engine is None:
            calibration_data = preprocess_images(batch, (3, 224, 224))
            engine = build_engine(
                onnx_model_path,
                max_batch_size=batch_size,
                enable_fp16=True,
                enable_int8=False,
                calibration_data=calibration_data,
                engine_file_path=tenssort_path
            )
            if engine is None:
                raise RuntimeError("Failed to build the engine.")

        images = preprocess_images(batch, (3, 224, 224)) if calibration_data is None else calibration_data
        network_features = extract_features_batch(engine, images, batch_size=batch_size)
        cuda.Context.synchronize()

        for j in range(len(network_features)):
            query_vectors = np.tile(network_features[j], (feature_vectors.shape[0], 1))
            similarity_values = calculate_similarity_batch(query_vectors, feature_vectors, use_gpu=True)
            max_idx = np.argmax(similarity_values)
            sim_val = similarity_values[max_idx]
            _, path1 = batch[j]
            path2, class_id, _ = feat_data[max_idx]
            results.append([path1, path2, sim_val])

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    with open(output_path, 'w') as file:
        for item in sorted_results:
            file.write(f"{os.path.basename(item[0])},{os.path.basename(item[1])},{item[2]:.4f}\n")
    print(f"Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    try:
        data_path = "/kaggle/input/imagenet1kmediumtest-10k/test_10000/test_10000/*/*.JPEG"
        raw_data = glob.glob(data_path)
        data = []
        for path in raw_data:
            class_id = path.split("/")[-2]
            data.append([class_id, path])

        # Stratified split
        start_time = time.perf_counter()
        train_data, test_data = stratified_split_fixed_count(data, train_count=1000, test_count=5000, random_state=42)
        end_time = time.perf_counter()
        print(f"Stratified split took {end_time - start_time:.2f} seconds")

        # #Feature vector generation with time measurement
        # start_time = time.perf_counter()
        # onnx_model_path = "/kaggle/working/model.onnx" # necessary	
        # tenssort_path="/kaggle/working/model.engine" # necessary path
        # feat_vect_path = "/kaggle/working/feat_vec.pkl"
        # gen_feat_vec(train_data,tenssort_path,feat_vect_path,onnx_model_path)
        # end_time = time.perf_counter()
        # print(f"Feature vector generation took {end_time - start_time:.2f} seconds")
        
        # Query processing
        start_time = time.perf_counter()
        onnx_model_path = "/kaggle/working/model.onnx"
        feat_vect_path = "/kaggle/input/d/jhcnode/data00/feat_vec.pkl"
        tenssort_path = "/kaggle/input/d/jhcnode/data00/model.engine"
        output_path="/kaggle/working/output.csv"
        query_data(test_data, feat_vect_path, tenssort_path,output_path,onnx_model_path)
        end_time = time.perf_counter()
        print(f"Query processing took {end_time - start_time:.2f} seconds")
    finally:
        global_context.pop()
