# Fastest GPU kernels, written from scratch.

## Matrix Multiplication

Matrix multiplication of square bf16 matrices, accumulated in fp32.

```
N=4096
Kernel: 763 TFLOPs
cuBLAS: 716 TFLOPs

N=8192
Kernel: 808 TFLOPs
cuBLAS: 795 TFLOPs
```

Explanation in https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog

##### To run:
```
make matmul && out/matmul
```
Example kernels are in [`examples/matmul/`](https://github.com/pranjalssh/fast.cu/tree/main/examples/matmul) and orchestration is in [`matmul.cu`](https://github.com/pranjalssh/fast.cu/blob/main/matmul.cu)

## Sum reduction

We compute sum of 2^30 elements.

##### To run:
```
make sum && out/sum
```

```
Kernel: 3240.11 GB/s
cub Library: 3193 GB/s
```

Example kernels are in [`sum.cu`](https://github.com/pranjalssh/fast.cu/tree/main/sum.cu)
