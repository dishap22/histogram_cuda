[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q7lnW6We)

# GPU_Histogram Report

## Performance Summary

| Method                | Time (sec) |
|------------------------|------------|
| **CPU**               | 0.304793   |
| **Naive GPU**         | 0.168717   |
| **Shared Memory GPU** | 0.006726   |

## Optimizations Made

- **Naive GPU**:
  - Each thread processes one or more input elements and uses `atomicAdd()` to increment a bin in the **global histogram**.
  - Led to a **1.8x speedup** over the CPU implementation
  - Although this version runs in parallel, performance suffers because:
    - All threads contend for global memory access.
    - `atomicAdd()` calls to global memory are slow and serialized when multiple threads write to the same bin.

- **Shared Memory GPU**:
  - The kernel was redesigned to leverage **shared memory**:
    - Each thread block maintains a **private shared memory histogram**.
    - Threads within the block use `atomicAdd()` on this shared histogram.
    - After local computation, each thread in the block contributes to aggregating the shared histogram into the global histogram.
    - These threads are executed in warps (groups of 32 threads) which are the fundamental execution units on NVIDIA GPUs. While each thread operates independently, warp-based scheduling ensures efficient utilization of GPU resources and helps hide memory access latencies, leading to faster overall execution.
  - Also introduced **padding** to prevent cache line conflicts.

## Observations

- Shared memory GPU version achieved a **~45× speedup over CPU** and **~25× over naive GPU**.
- Local aggregation and minimized atomic operations led to a dramatic performance improvement.

---
