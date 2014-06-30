CXX      = g++
CXXFLAGS = -Wall -fPIC -O -g

MPICXX   = mpicxx
NVCC     = nvcc
NVFLAGS  = -O -g

AR	 = ar
LIBMISC	 = libdescrypt.a
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

lib: clean compile
	${AR} rv ${LIBMISC} *.o
	${RANLIB} ${LIBMISC}
	rm *.o
compile:
	${CXX} ${CXXFLAGS} ${HEADER} -c ${SOURCES}

all : ${ALL}

generator: Interface/DESGenerator.cpp
#	module purge && module load openmpi/gcc/1.6.5 && ${MPICXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	module purge && module load icc/13.1.1 impi/4.1.1.036 && ${MPICXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}

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
	module purge && module load cuda/5.5 && ${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${CP} $@  ${BINARY}

rungen: generator
	mpirun -np 4 ./$^ 4096 33554432 test

runcuda: gencuda
	./$^ 4096 262144 cuda

astyle:
	astyle --style=allman */*.cpp

creat:
	-mkdir Binary

clean:
	rm -f ${ALL} *.o DES_* *.txt
	-rm -rf Binary
