NVCC = nvcc 
NVCC_FLAGS = -arch=sm_86 -O3 --use_fast_math -Xcompiler "-fopenmp"
NVCC_LDFLAGS = -lcublas -lcublasLt

TARGETS = encoder_forward layernorm_forward matmul_forward

all: $(TARGETS)

matmul_forward: matmul_forward.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) $^ -o $@

layernorm_forward: layernorm_forward.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) $^ -o $@

encoder_forward: encoder_forward.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LDFLAGS) $^ -o $@

clean: 
	rm -f $(TARGETS)
