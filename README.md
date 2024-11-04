# Handwritten H100 matmul kernels

We will compute `C(column major) = A(row major) * B(column major)` for square size matrices(M=N=K) with BF16 type and FP32 accumulators.
We initialize elements by `normal_distribution(mean = 0, std_dev = 1)`.

We perform better for some sizes, at par for some, and worse for some :)

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



0 is baseline. 6, 7 are the best ones of the handwritten.

## Benchmarks

Run on H100 SXM.

N=2048*3: Kernel 7 is best.
```
KERNEL 0
Average elapsed time: (0.000627) s, performance: (  739.8) TFLOPS. size: (6144).

KERNEL 1
Average elapsed time: (0.014328) s, performance: (   32.4) TFLOPS. size: (6144).

KERNEL 2
Average elapsed time: (0.000972) s, performance: (  477.5) TFLOPS. size: (6144).

KERNEL 4
Average elapsed time: (0.000714) s, performance: (  649.8) TFLOPS. size: (6144).

KERNEL 5
Average elapsed time: (0.000739) s, performance: (  627.9) TFLOPS. size: (6144).

KERNEL 6
Average elapsed time: (0.000613) s, performance: (  756.2) TFLOPS. size: (6144).

***
KERNEL 7
Average elapsed time: (0.000603) s, performance: (  769.3) TFLOPS. size: (6144).
***
```

N=2048*4: Kernels 0/7 are similar.
```
KERNEL 0
Average elapsed time: (0.001385) s, performance: (  794.1) TFLOPS. size: (8192).

KERNEL 7
Average elapsed time: (0.001396) s, performance: (  787.5) TFLOPS. size: (8192).
```

N=2048*6: Kernel 0 is best.
```
***
KERNEL 0
Average elapsed time: (0.004555) s, performance: (  814.6) TFLOPS. size: (12288).
***

KERNEL 1
Average elapsed time: (0.111039) s, performance: (   33.4) TFLOPS. size: (12288).

KERNEL 2
Average elapsed time: (0.007472) s, performance: (  496.6) TFLOPS. size: (12288).

KERNEL 4
Average elapsed time: (0.006013) s, performance: (  617.1) TFLOPS. size: (12288).

KERNEL 5
Average elapsed time: (0.005571) s, performance: (  666.1) TFLOPS. size: (12288).

KERNEL 6
Average elapsed time: (0.004770) s, performance: (  777.9) TFLOPS. size: (12288).

KERNEL 7
Average elapsed time: (0.004879) s, performance: (  760.6) TFLOPS. size: (12288).
```

## More Things to do:
* I haven't been able to use TMA loads of MxK with K > 64. M can vary till 256, but tensor map fails to create for K > 64 for any value of M. Need to dig more into cutlass to see how to do it.
* profiler says global accesses only utilize 50% bus width :((
* Better Tile scheduling(one that can use all SMs?). Hilbert curves aren't better than current scheduling.
* In A100/H100 L2 cache is split into 2 parts, reading from the "nearer" partition can speed up memory accesses. Is this worth reverse engineering and trying?
* Try reducing L2 cache size using L2 setaside feature - and see if it reduces power usage of kernel.
* ...

