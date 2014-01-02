PROGS = common chainWalkContext generator verified sort crack

LIB = -lrt -lssl -lcrypto -O0 -g

all: ${PROGS}

common: Common.cpp
	g++ -c $^ $(LIB)

chainWalkContext: ChainWalkContext.cpp
	g++ -c $^ $(LIB)

generator: Generate.cpp Common.cpp ChainWalkContext.cpp
	g++ $^ -o $@ ${LIB} -ldl

verified: common
	g++ $(CXXFLAGS) Public.cpp ChainWalkContext.cpp HashAlgorithm.cpp HashRoutine.cpp RainbowTableDump.cpp -lssl -o rtdump

sort: common
	g++ $(CXXFLAGS) Public.cpp RainbowTableSort.cpp -o rtsort

crack: common
	g++ $(CXXFLAGS) Public.cpp ChainWalkContext.cpp HashAlgorithm.cpp HashRoutine.cpp HashSet.cpp MemoryPool.cpp ChainWalkSet.cpp CrackEngine.cpp RainbowCrack.cpp -lssl -o rcrack
   
clean: common
	rm -f ${PROGS}