# CUDA GPU Performance & Memory Optimization

A hands-on study of CUDA kernel performance, GPU memory hierarchy, warp divergence, memory latency, bandwidth, constant-memory access patterns, and global-memory coalescing.

## Notebook

The accompanying Google Colab notebook contains the **commands, CUDA implementations, experiments, and reasoning behind the experiments**, excluding the generated outputs.

**[Open the CUDA Project Notebook](https://colab.research.google.com/drive/1mhdkjYhd5YjaT68HG6mMJrSiX1ArrIcY?usp=sharing)**

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

