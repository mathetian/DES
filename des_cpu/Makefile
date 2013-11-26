cc = mpicc

main: des.c
	$(cc) $^ -o $@ -lrt -lssl -lcrypto -O0 -g -std=gnu99
des_os: des_os.c
	$(cc) $^ -o $@ -lrt -lssl -lcrypto -O0 -g -std=gnu99
subrun: main
	mpirun -np 4 ./main
subrunos: des_os
	mpirun -np 4 ./des_os
clean:
	rm -rf main *.o des_os