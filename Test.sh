# Test 1

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

# Test 2

# mpirun -np 4 ./generator des 4096 262144 test
# ./cuda des 4096 262144 cuda

# mpirun -np 4 ./generator md5 4096 262144 test
# ./cuda md5 4096 262144 cuda

# mpirun -np 4 ./generator sha1 4096 262144 test
# ./cuda sha1 4096 262144 cuda

mpirun -np 4 ./generator hmac 4096 262144 test
./cuda hmac 4096 262144 cuda