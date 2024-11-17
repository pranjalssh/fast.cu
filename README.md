# Handwritten H100 matmul kernels

We will compute `C(column major) = A(row major) * B(column major)` for square size matrices(M=N=K) with BF16 type and FP32 accumulators.
We initialize elements by `normal_distribution(mean = 0, std_dev = 1)`.

For N=4096, we are 9% faster than cuBLAS
For N=8192, we are 1.5% faster than cuBLAS

## To run:
```
make matmul && out/matmul
```

Example kernels are in [`examples/matmul/`](https://github.com/pranjalssh/fast.cu/tree/main/examples/matmul) and orchestration in [`matmul.cu`](https://github.com/pranjalssh/fast.cu/blob/main/matmul.cu)

## Kernel details

* Kernel 0:
    * Simple cuBLAS call
* Kernel 1:
    * Tiled matmul(GMEM -> SMEM -> RMEM) without tensor cores/double buffering from [Simon's blog](https://siboehm.com/articles/22/CUDA-MMM).
* Kernel 2:
    * One warpgroup per ThreadBlock doing TMA-loads + WGMMA + Stores. Both A, B are kept in SMEM for the WGMMA call. We reach occupancy of 3.
    * We use 192x192 ThreadBlock size with loads of 192x64 from A/B. m64n192k16 WGMMA instruction is used.
    * Another option is to store C in row-major, which allows us to vectorize writes(2 bf16 at a time) - but performance is almost same.
    * We use __stwt to skip caching stores of output, making more room for A/B in L2 cache.
* Kernel 3:
    * Debugging Only
* Kernel 4:
    * Kernel 2 + Producer/Consumer pattern + Buffering of A/B from SMEM.
    * We use 3 consumer warpgroups with same hyperparameters as above. 3 consumers compute different outputs across m-dimension. 192 -> 3x64.
    * `setmaxnreg` PTX command is used to increase/reduce register count.
    * We use cuda::barrier for synchronization(what cuda c++ programming guide recommends)
* Kernel 5:
    * Kernel 4 + Persistent kernels.
    * Use 128 SMs and schedule thread blocks in a 16x8 grid at a time. Move grid horizontally till it reaches end, then next 16 rows and so on...
    * Using 132 SMs is slightly slower...
* Kernel 6:
    * Kernel 5 + mbarrier instead of cuda::barrier + using parity api instead of arrival tokens. This is 20% faster(see ptx guide for api details)
* Kernel 7:
    * Kernel 6 + Thread Block clusters. We group two vertically consecutive(along M-dimension) ThreadBlock into a cluster, and use TMA-multicast to broadcast from A's gmem -> smem.
    * Grouping by m-dimension is better than n-dimension(??).
    * We also switch from 3->2 consumer warpgroups: 128x256 ThreadBlock size with m64n256k16 WGMMA. Using more warpgroups is harmful(maybe because more synchronization is needed?)
    * Performance drops to 60% when using clusters of size > 2. Looks like mbarrier synchronization is too slow for this?
* Kernel 11:
    * Use hilbert curve for scheduling output blocks(better cache locality)
    * Use TMA Stores for writing results back to GMEM




0 is cuBLAS baseline. 11 is the best handwritten version

## Benchmarks

Run on H100 SXM.

N=2048*4: Kernel 11 is best(808 TFLOPs) 1.5% faster than cuBLAS
```
KERNEL 0
Average elapsed time: (0.001378) s, performance: (  797.8) TFLOPS. size: (8192).

KERNEL 1
Average elapsed time: (0.033970) s, performance: (   32.4) TFLOPS. size: (8192).

KERNEL 2
Average elapsed time: (0.002251) s, performance: (  488.5) TFLOPS. size: (8192).

KERNEL 4
Average elapsed time: (0.001731) s, performance: (  635.4) TFLOPS. size: (8192).

KERNEL 5
Average elapsed time: (0.001670) s, performance: (  658.4) TFLOPS. size: (8192).

KERNEL 6
Average elapsed time: (0.001414) s, performance: (  777.5) TFLOPS. size: (8192).

KERNEL 7
Average elapsed time: (0.001376) s, performance: (  799.1) TFLOPS. size: (8192).

KERNEL 11
Average elapsed time: (0.001361) s, performance: (  807.7) TFLOPS. size: (8192).
```

N=2048*3: Kernel 11 is best(794.5 TFLOPs) 7% faster than cuBLAS
KERNEL 0
Average elapsed time: (0.000626) s, performance: (  741.1) TFLOPS. size: (6144).

KERNEL 1
Average elapsed time: (0.014438) s, performance: (   32.1) TFLOPS. size: (6144).

KERNEL 2
Average elapsed time: (0.000973) s, performance: (  476.6) TFLOPS. size: (6144).

KERNEL 4
Average elapsed time: (0.000712) s, performance: (  651.2) TFLOPS. size: (6144).

KERNEL 5
Average elapsed time: (0.000741) s, performance: (  625.8) TFLOPS. size: (6144).

KERNEL 6
Average elapsed time: (0.000605) s, performance: (  766.3) TFLOPS. size: (6144).

KERNEL 7
Average elapsed time: (0.000599) s, performance: (  774.4) TFLOPS. size: (6144).

KERNEL 11
Average elapsed time: (0.000584) s, performance: (  794.5) TFLOPS. size: (6144).
```


## More Ideas to try:
* Better Tile scheduling. one that can use all SMs
* In A100/H100 L2 cache is split into 2 parts, reading from the "nearer" partition can speed up memory accesses. Is this worth reverse engineering and trying?
* Try reducing L2 cache size using L2 setaside feature - and see if it reduces power usage of kernel.
* Try vectorizing shared memory stores using __shfl operations

