PROGS = generator verified sort crack

LIB = -lrt -lssl -lcrypto -ldl -O0 -g -lcurand


all: ${PROGS}

generator: Generate.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB} 

verified: Verified.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB}

sort: SortPreCalculate.cpp Common.cpp
	g++ $^ -o $@ ${LIB}
	
crack: DESCrack.cpp Common.cpp ChainWalkContext.cpp CipherSet.cpp CrackEngine.cpp MemoryPool.cpp
	g++ $^ -o $@ ${LIB}

nv = nvcc

gencuda: DESCuda.cu
	$(nv) $^ -o $@  ${LIB}

clean:
	rm -f ${PROGS} DES_* *.txt