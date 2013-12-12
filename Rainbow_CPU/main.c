#include "common.h"

int main(int argc, char*argv[])
{
    int numproc,rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    desCrypt(rank,numproc);      
    MPI_Finalize();
    return 0;
}
