## GPU: Graphics Processing Unit
While a CPU is designed for sequential, complex tasks — it has a few powerful cores (typically 8–32) optimized for low-latency, branchy logic like running an OS or a web server, a GPU is designed for parallel, repetitive tasks — it has thousands of smaller cores (e.g. NVIDIA's H100 has ~16,000 CUDA cores) all working simultaneously.

**Why Parallelism Transfers Beyond Graphics?**
Rendering a frame is fundamentally a math problem: transform millions of pixels using the same operations simultaneously. It turns out many non-graphics problems have the exact same shape:
| **Task** | **What it actually is** |
| --- | --- |
| Training a neural network | Multiply huge matrices, millions of times |
| Cryptocurrency mining | Hash the same function billions of times |
| Weather modeling | Solve differential equations over a grid simultaneously |

All of these share one trait: **do the same operation on massive amounts of data at once** — which is exactly what a GPU is built for.

**The Key Technical Reasons**
1. SIMD Architecture (Single Instruction, Multiple Data)
A GPU issues one instruction that executes across thousands of data points simultaneously. Perfect for matrix and vector math.
2. High Memory Bandwidth
GPUs have extremely fast memory buses (e.g. H100: ~3.35 TB/s vs a CPU's ~100 GB/s). When you're crunching billions of numbers, feeding data fast matters enormously.
3. Tensor Cores (Modern GPUs)
NVIDIA added dedicated hardware just for matrix multiplication — the core operation in deep learning — making them orders of magnitude faster than a CPU for AI workloads.
4. CUDA / OpenCL Ecosystems
NVIDIA's CUDA (2007) was the turning point — it gave developers a programming model to run general-purpose code on a GPU without thinking about graphics at all. This birthed the term GPGPU (General-Purpose GPU computing).

**CUDA**: Compute Unified Device Architecture
**OpenCL**: Open Computing Language. It is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators. 

**Back to Main Note**
[MainNote](MainNote.md)