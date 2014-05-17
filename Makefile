CXX      = g++
CXXFLAGS = -Wall -fPIC -O -g

MPICXX   = mpic++
NVCC     = nvcc
NVFLAGS  = -O -g

AR	     = ar
LIBMISC	 = libdescrypt.so
RANLIB   = ranlib
RM       = rm
MV       = mv
CP       = cp

SOURCES = Common/*.cpp
HEADER  = -I./Include
BINARY  = Binary

LIB = -lrt -lssl -lcrypto -ldl -L. -ldescrypt
LIB1 = -lrt -lssl -lcrypto -ldl
ALL = generator verified sort crack gencuda

lib: compile
	${CXX} -shared *.o ${LIB1} -o ${LIBMISC}
	${CP}  ${LIBMISC} ${BINARY}
	${RM} *.o

compile:
	${CXX} ${CXXFLAGS} ${HEADER} -c ${SOURCES}

all : ${ALL}

generator: Interface/DESGenerator.cpp
	${MPICXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB} 
	${CP} $@  ${BINARY}

verified: Interface/DESVerified.cpp
	${CXX} ${CXXFLAGS} ${HEADER} ${LIB} $^ -o $@ ${LIB}
	${CP} $@  ${BINARY}

sort: Interface/DESSort.cpp
	${CXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${CP} $@  ${BINARY}

crack: Interface/DESCrack.cpp
	${CXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${CP} $@  ${BINARY}

gencuda: Interface/DESCuda.cu
	${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${CP} $@  ${BINARY}

rungen: generator
	mpirun -np 4 ./$^ 8192 8388608 test

runcuda: gencuda
	./$^ 1024 250000 test

astyle:
	astyle --style=allman */*.cpp

creat:
	-mkdir Binary

clean:
	rm -f ${ALL} *.o DES_* *.a *.txt
	-rm -rf Binary