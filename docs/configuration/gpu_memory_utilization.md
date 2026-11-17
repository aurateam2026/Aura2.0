<a id="gpu-memory-calculation-and-configuration"></a>

# Accelerator Memory Calculation and Configuration

This guide explains how to calculate accelerator memory requirements and properly configure `gpu_memory_utilization` for vLLM-Omni stages.

## Overview

`gpu_memory_utilization` is a critical parameter that controls how much accelerator memory each stage can use. It's specified as a fraction between 0.0 and 1.0, where:

- `0.8` means 80% of the accelerator's total memory
- `1.0` means 100% of the accelerator's total memory (not recommended, leaves no buffer)

## How Memory is Calculated

### Memory Allocation Formula

For each stage, vLLM-Omni calculates the requested memory as:

```text
requested_memory = total_gpu_memory × gpu_memory_utilization
```

On discrete accelerators, when other stages already occupy the same accelerator, Omni caps this requested budget:

```text
requested_memory = min(requested_memory, free_memory)
```

The worker logs a warning when it caps the budget. Model and runtime allocations
are subtracted from that budget when sizing the KV cache. Initialization can
still fail if the remaining budget cannot hold the required cache. Raising
`gpu_memory_utilization` does not increase a budget already capped to free memory.

On integrated accelerators, accelerator allocations share memory with the host OS and other
processes. Omni preserves upstream startup validation: if the requested budget
exceeds free memory, initialization raises `ValueError` instead of capping the
budget. Reduce `gpu_memory_utilization` for each stage and leave headroom for the
host, other stages, and runtime allocations. A budget equal to free memory passes
this check, but the check itself does not reserve any headroom.

### Memory Components

The total memory used by a stage includes:

1. **Model Weights**: The size of the model parameters loaded on the accelerator
2. **KV Cache**: Memory for storing key-value cache during generation
3. **Activation Memory**: Temporary memory for intermediate computations
4. **System Overhead**: Memory used by accelerator runtime, PyTorch, and other system components
5. **Non-Torch Memory**: Memory allocated outside of PyTorch (e.g., accelerator graphs)

When a stage captures accelerator graphs, the pool the runner captures through its
Accelerator runtime-graph dispatcher is profiled during startup and subtracted from the KV cache
budget, so `gpu_memory_utilization` covers those graphs as well. Set
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to skip that profiling pass and leave the pool unreserved. A stage tuned
before this reserve existed keeps its previous KV cache size by raising
`gpu_memory_utilization` by the graph pool's share of the device. The startup log
prints the value to use when the raised budget still fits in the memory free at
startup, and otherwise says that no value restores the previous size, for example
when co-located stages already hold part of the device. On accelerator runtime and ROCm, the reserve also covers the
talker MTP graphs, which are profiled with the rest, and, on stages that size a KV
cache, the graphs of models that implement both `capture_auxiliary_graphs` and
`release_auxiliary_graphs`, which are captured once while profiling, released, and
captured again at warmup.
Graphs a model captures on its own outside these paths, or on a stage without a KV
cache, are not part of it, and stages using them need those pools free on top of
their utilization.

### Example Calculation

For an accelerator with 80GB total memory:

- `gpu_memory_utilization: 0.8` → 64GB available for the stage
- `gpu_memory_utilization: 0.6` → 48GB available for the stage
- `gpu_memory_utilization: 0.15` → 12GB available for the stage

## Setting Up `gpu_memory_utilization`

<a id="step-1-determine-gpu-memory"></a>

### Step 1: Determine accelerator Memory

First, check your accelerator's total memory:

```bash
# Using accelerator vendor-smi
nvidia-smi --query-gpu=memory.total --format=csv

# Or using Python
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

### Step 2: Estimate Model Memory Requirements

#### For Autoregressive (AR) Stages

AR stages typically need more memory due to:

- Large model weights
- KV cache for attention
- Activation buffers

#### For Diffusion/Generation Stages

Diffusion stages (like code2wav) typically need less memory:

- Smaller model components
- Different memory access patterns

**Typical values:**

- `0.1 - 0.3` for most diffusion stages

### Step 3: Consider Multi-Stage Scenarios

When multiple stages share the same accelerator, you must ensure the sum of their `gpu_memory_utilization` values doesn't exceed 1.0.

<a id="example-two-stages-on-gpu-0"></a>

#### Example: Two stages on accelerator 0

```yaml
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.6  # Uses 60% of GPU 0

  - stage_id: 1
    devices: "0"
    gpu_memory_utilization: 0.3  # Uses 30% of GPU 0
    # Total: 90% of accelerator 0; check whether the remaining 10% is enough headroom
```

**Important:** Stages on discrete accelerators with separate memory pools can budget
independently, while leaving headroom on each accelerator. Devices that share a host memory
pool must also account for their combined allocations and host memory use; separate
device IDs do not make those budgets independent.

### Step 4: Account for Tensor Parallelism

When using `tensor_parallel_size > 1`, the model is split across multiple accelerators, so each accelerator needs less memory.

#### Example: 2-way tensor parallelism

```yaml
stages:
  - stage_id: 0
    devices: "0,1"  # Uses both GPUs
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6  # 60% per GPU
    # Model is split, so each accelerator uses ~30% of model memory
```

## Examples

### Qwen3-Omni-MoE on 2x H100-80GB

```yaml
base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml

stages:
  - stage_id: 0  # Thinker stage with TP=2
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6  # 48GB per GPU

  - stage_id: 1  # Talker stage
    devices: "1"
    gpu_memory_utilization: 0.3  # 24GB on GPU 1

  - stage_id: 2  # Code2Wav stage
    devices: "0"
    gpu_memory_utilization: 0.1  # 8GB on GPU 0
```

**Note:** Stage 0 uses accelerators 0 and 1, so the combined utilization is `0.7` on
Accelerator 0 and `0.9` on accelerator 1. Keep the sum of all resident stages below `1.0` on
each device.

## Troubleshooting

<a id="error-free-memory-is-less-than-desired-gpu-memory-utilization"></a>

### Error: "Free memory is less than desired accelerator memory utilization"

This means the accelerator doesn't have enough free memory when the stage starts.
Integrated accelerators raise this error when the requested budget exceeds free memory;
discrete accelerators use the capped budget described above.

**Solutions:**

1. Free up memory by closing other processes
2. Reduce `gpu_memory_utilization` for this stage
3. Use an accelerator with more memory
4. Move the stage to a different accelerator

### Error: OOM during inference

The stage initialized but ran out of memory during processing.

**Solutions:**

1. Reduce `max_num_batched_tokens`
2. Reduce `max_num_seqs` in engine_args
3. Lower `gpu_memory_utilization` slightly
4. Enable quantization if supported

### Memory Not Fully Utilized

If you see low memory usage, you can:

1. Increase `gpu_memory_utilization` to allow larger KV cache
2. Increase `max_num_batched_tokens` for better batching
3. Check if other stages are limiting throughput

## Useful formula for Memory Calculation

### KV Cache Memory

The KV cache size depends on:

- Number of sequences in batch
- Sequence length (prompt + generation)
- Model hidden size
- Number of attention heads
- Number of layers

approximate Formula:

```text
kv_cache_memory ≈ batch_size × seq_len × hidden_size × num_layers × 2 × dtype_size
```

2 for k & v

### Model Weight Memory

```text
model_memory ≈ num_parameters × dtype_size
```

For example:

- 7B parameters in FP16: ~14GB
- 7B parameters in FP32: ~28GB
- 7B parameters in INT8: ~7GB

### Activation Memory

Activation memory is typically smaller but varies with:

- Batch size
- Sequence length
- Model architecture

It's usually 10-30% of model weight memory during inference.
