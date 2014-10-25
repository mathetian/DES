CXX      = g++
CXXFLAGS = -Wall -fPIC -O -g

MPICXX   = mpicxx
NVCC     = nvcc
NVFLAGS  = -O -g

AR	     = ar
LIBMISC	 = libdescrypt.a
RANLIB   = ranlib
RM       = rm
MV       = mv

SOURCES = Common/*.cpp
HEADER  = -I./Include
BINARY  = Binary

LIB = -L. -L/usr/local/ssl/lib -lrt -ldescrypt -lssl -lcrypto -ldl 

ALL = generator verified sort crack cuda crackcuda

lib: clean compile
	${AR} rv ${LIBMISC} *.o
	${RANLIB} ${LIBMISC}
	rm *.o

compile:
	${CXX} ${CXXFLAGS} ${HEADER} -c ${SOURCES}

all: creat ${ALL}

generator: Interface/RainbowGenerator.cpp
	#module purge && module load icc/13.1.1 impi/4.1.1.036 && ${MPICXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MPICXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

verified: Interface/RainbowVerified.cpp
	${CXX} ${CXXFLAGS} ${HEADER} ${LIB} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

sort: Interface/RainbowSort.cpp
	${CXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

crack: Interface/RainbowCrack.cpp
	${CXX} ${CXXFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

cuda: Interface/CUDA/RainbowCUDA.cu
	#module purge && module load cuda/5.5 && ${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

crackcuda: Interface/CUDA/RainbowCrackCUDA.cu
	#module purge && module load cuda/5.5 && ${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${NVCC} ${NVFLAGS} ${HEADER} $^ -o $@ ${LIB}
	${MV} $@  ${BINARY}

rungen: generator
	mpirun -np 4 ./$^ des 4096 65536 test

runcuda: gencuda
	./$^ 4096 65536 cuda

astyle:
	astyle --style=allman */*.cpp

creat:
	-mkdir Binary

clean:
	rm -f ${ALL} *.o DES_* *.txt
	-rm -rf Binary