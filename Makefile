PROGS = common chainWalkContext generator verified sort crack

LIB = -lrt -lssl -lcrypto -ldl -O0 -g

all: ${PROGS}

generator: Generate.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB} 

verified: Verified.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB}

sort: SortPreCalculate.cpp Common.cpp
	g++ $^ -o $@ ${LIB}
	
crack: common
	g++ $(CXXFLAGS) Public.cpp ChainWalkContext.cpp HashAlgorithm.cpp HashRoutine.cpp HashSet.cpp MemoryPool.cpp ChainWalkSet.cpp CrackEngine.cpp RainbowCrack.cpp -lssl -o rcrack
   
clean: common
	rm -f ${PROGS}