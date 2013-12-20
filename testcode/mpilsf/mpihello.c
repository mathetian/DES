#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <netdb.h>

#define MAX_HOSTNAME_LENGTH 200

int main(int argc, char *argv[])
{
    char hostname[MAX_HOSTNAME_LENGTH];
    int pid, numprocs, rank, rc;

    rc = MPI_Init(&argc, &argv);
    /* Get the number of processes and the rank of this process */
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* let's see who we are to the "outside world" - what host and what PID */
    gethostname(hostname, MAX_HOSTNAME_LENGTH);
    pid = getpid();
    printf("Rank %d of %d has pid %5d on %s\n", rank, numprocs, pid, hostname);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
