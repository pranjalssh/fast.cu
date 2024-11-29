NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
NVCC_INCLUDES = -I/usr/local/cuda-12.6/include
NVCC_LDLIBS =
NCLL_INCUDES =
OUT_DIR = out

GPU_COMPUTE_CAPABILITY=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
# if GPU_COMPUTE_CAPABILITY = 90, set it to 90a
GPU_COMPUTE_CAPABILITY := $(if $(findstring 90,$(GPU_COMPUTE_CAPABILITY)),90a,$(GPU_COMPUTE_CAPABILITY))

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
# NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
# NVCC_FLAGS += --ptxas-options=-v #,--register-usage-level=10
# NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo $(NVCC_INCLUDES) $(NVCC_LDLIBS)

sum: sum.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

sumprofile: sum
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

matmul: matmul.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmulprofile: matmul
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

clean:
	rm $(OUT_DIR)/*
