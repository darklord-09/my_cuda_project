# CUDA GPU Performance & Memory Optimization

A hands-on study of CUDA kernel performance, GPU memory hierarchy, warp divergence, memory latency, bandwidth, constant-memory access patterns, and global-memory coalescing.

## Notebook

The accompanying Google Colab notebook contains the **commands, CUDA implementations, experiments, and reasoning behind the experiments**, excluding the generated outputs.

**[Open the CUDA Project Notebook](https://colab.research.google.com/drive/1qS08rgq1V0-E6YQskHhGY7OUS6bcCnKI)**

---

## Week 1 — Warp Divergence & Control Flow

* Implemented two basic CUDA vector-addition kernels:

  * A baseline kernel with uniform control flow.
  * A kernel containing a thread-ID-dependent branching decision.
* Measured the performance impact of **warp divergence** on kernel execution time.
* Used profiling tools to examine differences in **instruction flow and execution behavior** between the two kernels.
* Studied how divergent branches within a warp affect instruction execution and overall performance.

## Week 2 — GPU Memory Hierarchy & Memory Performance

### GPU Memory Hierarchy

Studied the characteristics and roles of major GPU memory spaces:

* Registers
* L1 Cache
* Shared Memory
* L2 Cache
* Global Memory
* Constant Memory

### Global & Shared Memory Latency

* Implemented a simple vector-copy kernel to experimentally measure **global-memory latency in GPU cycles** using a custom harness.
* Implemented a corresponding vector-copy kernel targeting **shared memory** to study the difference in memory-access latency.

### Constant Memory — Broadcast vs. Scatter

* Implemented kernels to investigate **broadcast and scattered access patterns** in constant memory.
* Measured both access patterns using a custom performance harness.
* Investigated unexpected experimental results rather than assuming the initial hypothesis was correct.
* Built an additional experimental kernel to isolate the underlying behavior and performed a deeper analysis of the observed results.

### Global Memory Bandwidth

* Implemented a simple memory-copy kernel performing **one global-memory read and one global-memory write per element**.
* Used the kernel to experimentally measure achievable **global-memory bandwidth**.

### Matrix Transpose & Memory Coalescing

* Implemented a naive CUDA matrix-transpose kernel.
* Studied the impact of **non-coalesced global-memory accesses**, particularly the strided store pattern introduced during transposition.
* Measured kernel performance using a custom harness to establish a baseline for comparison with optimized transpose implementations.

## Week 3 — GPU Memory Hierarchy & Memory Performance continued

* Fixed the transpose kernel with shared-memory tiling to restore coalesced access
* Handled shared-memory bank conflicts (the padding trick)
* Parallel reduction kernel, progressing through the standard stages: naive → shared memory → warp-shuffle-based → divergence-free


## Week 4 - Roofline model
* Learn the roofline model: how to tell whether a kernel is compute-bound or memory-bound, and why that determines which optimization is worth trying next

* Implemented for every kernel made till week 4

## Week 5 - Naive SGEMM 

* Naive SGEMM (one thread per output element)
* Fix global-memory coalescing
* Correctness harness against cuBLAS output (not just against itself) — get this right now, since every later stage depends on trusting your correctness check
* Record baseline GFLOPS and % of cuBLAS

## Week 6 - Tiled SGEMM

* Shared-memory tiled SGEMM: block A and B into shared memory tiles to cut global-memory traffic
* Re-benchmark, record the jump in % of cuBLAS


