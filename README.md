# CUDA, TensorRT, and PyTorch-based Feature Extraction and Similarity Matching

This script integrates NVIDIA TensorRT, PyTorch, and CuPy to optimize feature extraction and similarity matching for image datasets. Special emphasis has been placed on **feature extraction optimization** and **efficient use of TensorRT for accelerated inference**. The entry point for the script is `run.py`.

---

## **Key Optimizations**

### 1. **Feature Extraction Optimization**
- **Batch Processing**:
  - Processes images in batches to maximize GPU utilization while avoiding memory bottlenecks.
  - Example: Feature extraction is handled in batches of a configurable size (e.g., 200 images at a time).
- **Parallel Preprocessing**:
  - Image loading and preprocessing are parallelized using `ThreadPoolExecutor` for faster input pipeline.
  - Preprocessing includes resizing, normalization, and tensor conversion, optimized for GPU inference.

```python
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
```

- **CUDA Memory Management**:
  - Efficiently allocates and deallocates GPU memory for input and output tensors during batch processing.
  - Synchronizes CUDA contexts to prevent race conditions.

```python
def extract_features_batch(engine, images, batch_size):
    ...
    d_input = cuda.mem_alloc(input_host.nbytes)
    d_output = cuda.mem_alloc(batch_size * 2048 * np.float32().nbytes)
    ...
    cuda.memcpy_htod(d_input, input_host)
    context.execute_v2(bindings)
    ...
    cuda.memcpy_dtoh(output_host, d_output)
    d_input.free()
    d_output.free()
```

---

### 2. **TensorRT Integration**
TensorRT is utilized to significantly accelerate the feature extraction process by optimizing the ONNX model for GPU execution.

#### **Steps for TensorRT Integration:**
1. **ONNX Model Conversion**:
   - Converts a pre-trained PyTorch ResNet50 model to ONNX format with dynamic batch support.
   - TensorRT can then optimize this ONNX model for inference.

```python
torch.onnx.export(
    model, dummy_input, onnx_model_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=13
)
```

2. **Building the TensorRT Engine**:
   - TensorRT optimizes the ONNX model for FP16 precision (or INT8 if calibration data is provided), leveraging GPU hardware acceleration.
   - If a serialized engine already exists, it is loaded directly to save time.

```python
def build_engine(onnx_file_path, max_batch_size=4, enable_fp16=True, enable_int8=False, ...):
    ...
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
    if enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if enable_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    ...
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    ...
```

3. **Inference with TensorRT**:
   - Once the TensorRT engine is built, feature extraction is performed directly on the GPU, reducing latency and maximizing throughput.

```python
context = engine.create_execution_context()
context.set_binding_shape(0, input_shape)
bindings = [int(d_input), int(d_output)]
context.execute_v2(bindings)
```

---

## **Main Execution in `run.py`**
1. **Stratified Data Splitting**:
   - Ensures balanced class distribution in training and testing datasets.
2. **Feature Extraction**:
   - Uses TensorRT-optimized inference to extract feature vectors from training images.
3. **Query Processing**:
   - Efficiently compares test images against pre-extracted feature vectors using cosine similarity.

```python
if __name__ == "__main__":
    ...
    # Query processing
    start_time = time.perf_counter()
    query_data(test_data, feat_vect_path, tenssort_path, output_path, onnx_model_path)
    end_time = time.perf_counter()
    print(f"Query processing took {end_time - start_time:.2f} seconds")
```

---

## **Performance Highlights**
- **TensorRT Optimizations**:
  - FP16 precision for faster inference while maintaining accuracy.
  - Serialized engine reduces initialization overhead for subsequent runs.
- **Feature Extraction**:
  - Batch-based processing leverages GPU throughput effectively.
  - CuPy accelerates similarity calculations for large datasets.

---

## **Usage**
For ease of use, the script can be executed in a Kaggle environment, which provides:
- Pre-installed dependencies (e.g., TensorRT, PyCUDA, PyTorch).
- Access to GPUs without requiring additional setup.
- A pre-configured environment for running the `run.py` script seamlessly.

### Steps to Execute in Kaggle:
1. Upload your script and dataset to a Kaggle notebook.
2. Ensure the required paths (e.g., ONNX model, TensorRT engine, and dataset paths) are correctly configured in the script.
3. Execute the script in a GPU-enabled Kaggle notebook.

### Command to Run:
```bash
python run.py
```

---

## **Dependencies**
- PyTorch
- torchvision
- NVIDIA TensorRT
- PyCUDA
- PIL
- NumPy
- CuPy
- scikit-learn

