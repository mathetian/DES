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

# mpirun -np 4 ./generator hmac 4096 262144 test
# ./cuda hmac 4096 262144 cuda

# Test 3 (Crack)
# mpirun -np 4 ./generator des 4096 262144 test
# ./sort 4 des_4096-262144_test
# ./verified des des_4096-262144_test 4096
# ./generator des test
# ./crack des file des_4096-262144_test Test.txt

mpirun -np 4 ./generator md5 4096 262144 test
./sort 4 md5_4096-262144_test
./verified md5 md5_4096-262144_test 4096
./generator md5 test
./crack md5 file md5_4096-262144_test Test.txt