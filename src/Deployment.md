# CPU Deployment with Low Latency


## **Overview**
Our system is designed to provide **RAG-based question answering** and **genre classification** with low latency on CPU. It combines multiple models and a vector database:

### **1. Core Models**

- **Qwen LLM**  
  - Versions: **2.5B** and **1.5B**  
  - Main engine for retrieval-augmented generation (RAG) and instruction-following tasks  
  - Largest model in the system; has the most significant impact on performance and latency  

- **E5 Small Embedding Model**  
  - Generates embeddings for story chunks  
  - Enables **efficient retrieval** from Qdrant via HNSW and metadata filtering  

- **Sentence Transformer (SetFit)**  
  - Few-shot genre classification  
  - Lightweight and optimized for CPU  
  - Uses **contrastive learning** to handle limited labeled data  
### **2. Vector Database**

- **Qdrant**  
  - Stores story embeddings and metadata  
  - Supports **hybrid search** (vector similarity + metadata filtering)  
  - Provides fast approximate nearest neighbor (ANN) search via HNSW  
## **3. Deployment Goal**

- Efficient CPU-based inference for **both question answering and genre classification**  
- Maintain **low-latency response times** while using limited system resources  
- Focus on **Qwen** as the largest and most impactful model, optimizing its performance for CPU deployment  
### **4. Latency Baseline**

Before any optimization, the raw Qwen LLM exhibits **high inference latency**, making CPU deployment impractical:

**Latency Example (Qwen Before Optimization):**  

## Benchmark Results

| Metric                     | Value                |
|-----------------------------|--------------------|
| Average Latency             | 9.5657 seconds     |
| Average Generation Speed    | 42.13 tokens/sec   |
 

> As seen, unoptimized latency is too high for production use, highlighting the need for optimization strategies to achieve acceptable performance on CPU.
---
## CPU Optimization Techniques

Deploying large LLMs like **Qwen 2.5B** on CPU requires careful optimizations to reduce latency while preserving accuracy.  

| Technique | Description | Pros | Cons | Why We Chose / Notes |
|-----------|-------------|------|------|--------------------|
| **Quantization** | Reduces the precision of model weights from **FP32 to INT8/INT4**, lowering memory footprint and speeding up inference | - Dramatically reduces memory usage<br>- Improves CPU inference speed<br>- Minimal modification to the original model | - Slight accuracy degradation possible<br>- Some hardware may not fully support extreme low-bit precision | - Fits CPU constraints while maintaining usable model accuracy<br>- Reduces 2.5B Qwen model to practical size<br>- Meets real-time inference needs |
| **Pruning** | Removes weights/connections that contribute least to model output | - Smaller model size<br>- Faster inference with sparse computation | - Requires careful retraining<br>- Complex implementation<br>- Less compatible with LoRA | Can reduce memory and computation, but not chosen due to complexity and retraining overhead |
| **Distillation** | Trains a smaller “student” model to mimic outputs of a larger “teacher” | - Produces compact models<br>- Reduces memory and computation | - Training is time-consuming<br>- Requires additional or synthetic data<br>- May lose instruction-following capability | Good for low-resource deployment but requires extra training; not practical for our single-model  setup |
| **Other CPU Optimizations** | Low-level inference optimizations like fused kernels, batching, memory mapping | - Speeds up matrix multiplications<br>- Reduces overhead for multiple queries<br>- Can reduce peak memory | - Some methods may increase complexity<br>- Memory mapping may increase latency | Useful in combination with quantization but does not replace model-level optimization |

### Chosen Strategy

We selected **INT8/INT4 Quantization** for Qwen LLM because it provides the **best balance of latency, memory usage, and accuracy**.  

- Compatible with existing pipeline  
- Reduces large Qwen model to **practical CPU size**  
- Enables **real-time query responses** without GPU  
- Minimal accuracy loss while maintaining instruction-following behavior  

> Quantization allows our RAG pipeline and genre classifier to run efficiently on CPU while keeping performance practical.
### Quantization Comparison: INT8 vs INT4

We experimented with both **INT8** and **INT4** quantization for the Qwen LLM to optimize CPU inference.  

| Metric | INT8 | INT4 |
|--------|------|------|
| Average Latency | 4.0541 seconds | 4.0314 seconds |
| Average Generation Speed | 97.66 tokens/sec | 98.50 tokens/sec |
| Maximum Context Length (tokens) | 4k |  512  |
| Accuracy Impact | Minimal | Noticeable drop |
| Memory Usage | Moderate | Lowest |
| Practicality | High (chosen) | Medium |


### Decision

- **INT8** was selected because it maintains **Acceptable context length**, preserves **model accuracy**, and provides **fast CPU inference**.  
- **INT4** offers slightly faster generation and lower memory usage, but the **loss in context length** and **accuracy drop** made it less suitable.  

> This trade-off ensures our RAG system remain **robust and efficient on CPU**, while still benefiting from quantization.
### Optimization: Llama.cpp Migration

To further improve CPU inference speed and reduce overhead, we migrated our LLM from the standard **Transformers library** to **llama.cpp** using **llama-cpp-python bindings**.

### Why Llama.cpp?

- Written in **C/C++**, designed specifically for **CPU execution**.  
- Avoids **PyTorch overhead** (Python interpreter, dynamic computation graph).  
- **Single C++ binary** or **Python bindings** required.  
- No need for **PyTorch, TensorFlow, or CUDA installation**.  
- Makes inference **faster and more predictable** on CPU hardware.  

### Benefits

- **Lower per-query latency** for chatbot and classification tasks.  
- **Better CPU utilization**, reducing wasted resources.  
- **Simpler deployment**, as dependencies are minimal.  
- **Consistent performance**, independent of Python runtime variations.  
- Works well in combination with **INT8 quantization**, maximizing CPU efficiency.
---  
## Latency Benchmarking & Measurement Methodology

To evaluate CPU performance, we conducted structured latency benchmarking for both:

-  **Chatbot (RAG + Qwen)**
-  **Genre Classification (SetFit)**

All measurements were performed on CPU hardware under consistent load conditions.


### Chatbot Latency (Qwen LLM)

### Benchmarking Methodology

- Ran **50 queries** (random + structured prompts).
- Included different query types:
  - Short factual questions
  - Long contextual queries
  - Multi-step reasoning prompts
- Measured:
  - Total response time per query
  - Min / Max / Average latency
- Used **LangSmith tracing** to track:
  - Retrieval time
  - Generation time
  - Total end-to-end latency

#### 🚨 Before Quantization (Full Qwen – FP32)

| Metric | Value |
|--------|-------|
| Average Latency | **35.06 sec** |
| Minimum Latency | 18.1 sec |
| Maximum Latency | 100.78 sec |

⚠️ This performance is **not suitable for production CPU deployment**.

The high latency is caused by:
- Full precision weights (FP32)
- PyTorch runtime overhead
- Large model size (1.5B parameters)

### After INT8 Quantization + llama.cpp Migration

| Metric | Value |
|--------|-------|
| Average Latency | **18.24 sec** |
| Minimum Latency | 7.01 sec |
| Maximum Latency | 40.7 sec |

✔ Benchmarked over **50 full pipeline queries**  
✔ Measured using **LangSmith tracing**  
✔ Includes retrieval + generation  

### 🔥 Improvement Summary

- ~50% reduction in average latency  
- Significantly reduced worst-case response time  
- More stable and predictable CPU performance  

### Genre Classification Latency (SetFit)

### Benchmarking Methodology

- Single-text inference
- Multiple genre inputs tested
- Measured pure model prediction time
- CPU-only execution

## 📈 Classification Results

| Metric | Value |
|--------|-------|
| Average Latency | **1.04 sec** |
| Minimum Latency | 0.043 sec |
| Maximum Latency | 3.17 sec |

### Observations

- Lightweight sentence transformer architecture
- Suitable for real-time classification use cases
- Stable CPU performance even under repeated calls

### Conclusion

- **Quantization + llama.cpp migration reduced chatbot latency by ~50%.**
- Classification is already efficient and suitable for CPU production.
- Latency measurements were validated using:
  - Multiple query types
  - 50-query benchmarking
  - LangSmith tracing
  - Min / Max / Average reporting
---
## Deployment Strategy & Resource Efficiency

### Deployment Overview

The system is deployed using **Streamlit** as a lightweight web interface serving:

-  RAG Chatbot (Qwen + Qdrant + E5-small)
-  Genre Classification (SetFit)

The backend runs fully on **CPU**, using:

- **INT8 quantized Qwen**
- **llama.cpp (C++ backend)**
- Lightweight embedding and classification models

### Architecture Flow

>Important: The full pipeline is initialized **once at application startup** and cached for the entire Streamlit session.

When Streamlit launches:

- The **RAG pipeline** is loaded
- The **Qwen model (INT8 via llama.cpp)** is initialized
- The **E5-small embedding model** is loaded
- The **Qdrant client and collection** are connected
- The **SetFit classifier** is loaded

All components are **initialized once and cached**, meaning they are **not reloaded per user query**.
### Why This Matters

- Eliminates repeated model loading overhead  
- Reduces per-request latency  
- Prevents unnecessary memory reallocation  
- Improves stability under multiple queries  
- Ensures consistent CPU utilization  

### Resource Efficiency

###   Memory Requirements

- Minimum RAM required: **~7 GB**
- Recommended RAM: **8–16 GB** for stable multi-query handling
- Memory usage includes:
  - Quantized Qwen model (INT8)
  - Qdrant vector index
  - Embedding + classification models

Quantization significantly reduces memory footprint compared to full FP32 models.
### Minimum CPU Recommendation:

- **4 physical cores (8 threads)** — functional but slower under concurrency  
- **8 cores recommended** for stable multi-user handling  

The system does not require GPU acceleration but benefits from modern multi-core CPUs.
### Balancing Performance & Resource Usage

We optimized performance while keeping hardware requirements minimal by:

- **INT8 Quantization** → reduces memory and improves speed  
- **llama.cpp migration** → removes PyTorch overhead  
- **Lightweight embedding model (E5-small)**   
- Controlled context length to prevent CPU overload  

This ensures acceptable latency while staying within modest CPU and memory limits.


### Scalability

- Stateless request handling  
- Models loaded once at startup   
- Qdrant supports scalable vector indexing  


### Reliability

- Preloaded models reduce runtime failures  
- Exception handling for retrieval and generation errors  
- Latency monitoring using LangSmith tracing  
- Controlled context size to prevent excessive CPU spikes  

### Maintainability

- Modular architecture (RAG, classifier, vector DB separated)  
- Easy to swap:
  - Quantization levels (INT8 / INT4)
  - LLM versions (1.5B ↔ 2.5B)
  - Embedding models  
- Minimal dependencies after migrating to llama.cpp  
- Clear separation between UI and inference logic  
## Further Optimization Possibilities (Stricter Latency Requirements)

If even lower latency were required beyond our current CPU deployment, several **advanced optimization techniques** could be explored. These techniques are more aggressive and may involve trade‑offs in accuracy, engineering complexity, or additional tooling, but they represent the state‑of‑the‑art in model inference optimization for CPU environments.

### Combined Pruning and Quantization (Quant‑Aware Pruning)

- **What:** Pruning removes redundant weights/connections that contribute little to model output, while quantization reduces precision of remaining weights. These can be applied together.  
- **Effect:** Reduces compute and memory usage simultaneously, often yielding **2–3x speedups** if carefully balanced.  
- **Trade‑offs:** Pruning can degrade accuracy if over‑applied; tuning is required.  
- **How:** Techniques such as sparse pruning combined with low‑bit quantization reduce both FLOPs and memory bandwidth needs. :contentReference[oaicite:0]{index=0}

### Optimized Execution Engines (ONNX / TVM / Custom Kernels)

- **What:** Convert models to an optimized runtime such as **ONNX Runtime** or compilers like **TVM** for production inference.  
- **Advantages:**  
  - Automatic **operator fusion**, **constant folding**, and graph optimization.  
  - Backends often implement **vectorized kernels** using SIMD instructions like **AVX2 / AVX‑512**, improving CPU throughput.  
  - Works across CPUs without full deep learning frameworks. 
- **Trade‑offs:** Additional conversion step, might require tuning per CPU architecture.

### SIMD & Kernel Level Optimization

- **What:** Modern CPUs support **SIMD (Single Instruction, Multiple Data)** instructions (e.g., AVX, AVX2, AVX‑512) that enable parallel arithmetic across multiple data points.  
- **Impact:** Using SIMD‑optimized kernels can significantly speed up dense matrix multiplication and attention mechanisms — key operations in transformers.  
- **Why It Helps:** Optimized kernels reduce branch overhead and improve instruction throughput without changing model logic.
### Theoretical Latency Limits

The theoretical lower bound for latency is determined by:

- **Memory bandwidth** (data movement between RAM and CPU)  
- **Model compute complexity** (FLOPs per token)  
- **Instruction parallelism** (SIMD capabilities)

With aggressive optimization stacks (pruning + quantization + optimized runtimes + SIMD kernels), it’s theoretically possible to reach **several× lower latency** than naive CPU inference — especially with careful KV caching and batching. However:

- Lower precision or pruning may hurt accuracy.  
- Advanced runtimes like ONNX / TVM require per‑architecture tuning.  
- The decode phase remains inherently sequential (token by token), limiting how far latency can be reduced.

Therefore, achieving sub‑second average latency on large LLMs is challenging without high‑performance CPUs or specialized accelerators, but stacked optimizations can continue to significantly improve real‑world response times.

---
## Resources
- **AlexNet (Foundations of Deep CNNs)** – Paper: [https://arxiv.org/abs/1510.00149?utm](https://arxiv.org/abs/1510.00149?utm)  
- **Deep Compression & Efficient Models** – Paper: [https://arxiv.org/abs/1802.05668?utm](https://arxiv.org/abs/1802.05668?utm)  
- **ML Advanced: Model Optimization (Quantization, Pruning, etc.)** – Tutorial: [https://www.aplab.academy/en/courses/ml-advanced/lessons/model-optimization?utm](https://www.aplab.academy/en/courses/ml-advanced/lessons/model-optimization?utm)  
- **llama.cpp – High Performance LLM Inference Engine** – GitHub: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
- **IBM Quantization Overview & Best Practices** – Industry Guide: [http://ibm.com/qa-ar/think/topics/quantization?utm](http://ibm.com/qa-ar/think/topics/quantization?utm)



