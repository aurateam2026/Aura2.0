# Parallelism Acceleration Guide

This guide covers the parallelism methods in vLLM-Omni for speeding up diffusion model inference and reducing per-device memory requirements.

## Supported Methods

| Method                                             | Description                                                                                                         |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **[Tensor Parallelism](tensor_parallel.md)**       | Shards DiT weights across accelerators to reduce per-accelerator memory                                                             |
| **[Sequence Parallelism](sequence_parallel.md)**   | Splits sequence dimension across accelerators (Ulysses-SP, Ring-Attention, or hybrid) for high-resolution images and videos |
| **[CFG-Parallel](cfg_parallel.md)**                | Runs CFG positive/negative branches on separate accelerators for ~1.8x speedup on guided generation                         |
| **[Pipeline Parallelism](pipeline_parallel.md)**   | Splits the denoising transformer block-wise across sequential accelerator stages to reduce per-accelerator model memory             |
| **[VAE Parallelism](vae_parallelism.md)**          | Distributes VAE decode spatially across accelerators to reduce peak VAE memory                                              |
| **[HSDP](hsdp.md)**                                | Shards full model weights via PyTorch FSDP2 to enable large-model inference on memory-constrained accelerators              |
| **[Expert Parallelism](expert_parallel.md)**       | Shards MoE expert blocks across accelerators for MoE models (e.g. HunyuanImage3.0)                                          |

See [Supported Models](../../diffusion_features.md#supported-models) for per-model compatibility.
