# ./generator des testcasegenerator
# ./test des TestCaseGenerator.txt
# rm *.txt

# ./generator md5 testcasegenerator
# ./test md5 TestCaseGenerator.txt
# rm *.txt

# ./generator sha1 testcasegenerator
# ./test sha1 TestCaseGenerator.txt
# rm *.txt

# ./generator hmac testcasegenerator
# ./test hmac TestCaseGenerator.txt
# rm *.txt

mpirun -np 4 ./generator hmac 4096 16384 test
#mpirun -np 4 ./generator hmac 4096 65536 test

./cuda hmac 4096 65536 cuda