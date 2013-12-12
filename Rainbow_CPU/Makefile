cc = mpicc
nv = nvcc
flags = -lrt -lssl -lcrypto -O0 -g

rainbow: rainbow.c main.c
	$(cc) $^ -o $@ $(flags)
run : rainbow
	mpirun -np 4 ./$^
clean:
	rm -rf basic advanced job.err job.out *.o rainbow *.in

