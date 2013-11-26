In this part, I will give an example which shows how to use mpi technology in SJTU's HPC.

-----------------------------------------------------------------------------------------------------------------

1. login in the HPC. Custom the login.sh. and 
	$ make login
	or
	$ ./login.sh

1. Load the modules and make the environment variables to take effect. (I don't how to write shell script to save it)

	$ source /lustre/utility/intel/composer_xe_2013.3.163/bin/compilervars.sh intel64
	$ source /lustre/utility/intel/mkl/bin/intel64/mklvars_intel64.sh
	$ source /lustre/utility/intel/impi/4.1.1.036/bin64/mpivars.sh
	$ which mpicc
	$ /lustre/utility/intel/impi/4.1.1.036/intel64/bin/mpicc
	which mpirun
	/lustre/utility/intel/impi/4.1.1.036/intel64/bin/mpirun

2. Compile the ```mpihello``` program.

	$ make

3. Start the MPI program with ```mpirun```. *This is for test purpose only!!! Please don't use this approach in production!!!*

	$ make subrun

5. For production use, you should submit your job with LSF. This is an example.

	$ make run
