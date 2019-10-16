CUDA_DIR ?= /Developer/NVIDIA/CUDA-10.1
NVCC := $(CUDA_DIR)/bin/nvcc

libfive-cuda-test: libfive-cuda.cu gpu_interval.hpp
	$(NVCC) -g -I. -I$(CUDA_DIR)/samples/common/inc -lineinfo --std=c++11 -Llibfive/build/libfive/src -lfive -Ilibfive/libfive/include -o $@ $<
	install_name_tool -change @rpath/libfive.dylib @executable_path/libfive/build/libfive/src/libfive.dylib $@
clean:
	rm libfive-cuda-test
