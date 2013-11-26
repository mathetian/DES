CC = mpicc

mpihello: mpihello.c
	$(CC) $^ -o $@
run:
	bsub < run.sh
subrun:
	mpirun -np 10 ./mpihello
clean:
	rm -f mpihello mpihello.o *.log *.out *.err *.nodes nodelist file*
login:
	./login.sh
.PHONY: clean
