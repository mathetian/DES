cc = mpicc
nv = nvcc
flags = -lrt -lssl -lcrypto -O0 -g
advanced: advanced.cu
	$(nv) $^ -o $@ $(flags) -lcurand
basic: basic.cu
	$(nv) -o $@ $^ $(flags) -lcurand
cuda:
	bsub < cuda.lsf
arun: advanced
	./$^
baiscrun: basic
	./$^
clean:
	rm -rf basic advanced job.err job.out *.o

