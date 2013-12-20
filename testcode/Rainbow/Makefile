cc = mpicc
nv = nvcc
flags = -lrt -lssl -lcrypto -O0 -g

rainbow: rainbow.cu
	$(nv) $^ -o $@ $(flags) -lcurand
clean:
	rm -rf basic advanced job.err job.out *.o rainbow *.in

