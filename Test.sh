# Test 1

# ./generator des rand
# ./test des des.txt
# rm *.txt

# ./generator md5 rand
# ./test md5 md5.txt
# rm *.txt

# ./generator sha1 rand
# ./test sha1 sha1.txt
# rm *.txt

# ./generator hmac rand
# ./test hmac hmac.txt
# rm *.txt

# Test 2

# mpirun -np 4 ./generator des 4096 65536 cpu
# ./cuda des 4096 262144 cuda

mpirun -np 4 ./generator md5 4096 65536 cpu
./cuda md5 4096 262144 cuda

# mpirun -np 4 ./generator sha1 4096 65536 cpu
# ./cuda sha1 4096 262144 cuda

# mpirun -np 4 ./generator hmac 4096 65536 cpu
# ./cuda hmac 4096 262144 cuda

# Test 3 (Crack)
# mpirun -np 4 ./generator des 4096 262144 test
# ./sort 4 des_4096-262144_test
# ./verified des des_4096-262144_test 4096
# ./generator des test
# ./crack des file des_4096-262144_test Test.txt

# mpirun -np 4 ./generator md5 4096 65536 test
# ./sort 4 md5_4096-65536_test
# ./verified md5 md5_4096-65536_test 4096
# ./generator md5 test
# ./crack md5 file md5_4096-65536_test Test.txt