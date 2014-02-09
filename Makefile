PROGS = generator verified sort crack gencuda

LIB = -lrt -lssl -lcrypto -ldl -O0 -g

all: ${PROGS}

generator: Generate.cpp Common.cpp ChainWalkContext.cpp
	mpic++ $^ -o $@ ${LIB} 

verified: Verified.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB}

sort: SortPreCalculate.cpp Common.cpp
	g++ $^ -o $@ ${LIB}
	
crack: DESCrack.cpp Common.cpp ChainWalkContext.cpp CipherSet.cpp CrackEngine.cpp MemoryPool.cpp
	g++ $^ -o $@ ${LIB}

rungen:
	mpirun -np 4 ./generator 8192 8388608 test

nv = nvcc

gencuda: DESCuda.cu
	$(nv) $^ -o $@  ${LIB}

clean:
	rm -f ${PROGS} DES_* *.txt test3 test4 test5 gencuda
