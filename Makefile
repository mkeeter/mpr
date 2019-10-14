CUDA_DIR ?= /Developer/NVIDIA/CUDA-10.1
NVCC := $(CUDA_DIR)/bin/nvcc

libfive-cuda-test: libfive-cuda.cu
	$(NVCC) -I$(CUDA_DIR)/samples/common/inc -Llibfive/build/libfive/src -lfive -Ilibfive/libfive/include -o $@ $^
clean:
	rm libfive-cuda-test
