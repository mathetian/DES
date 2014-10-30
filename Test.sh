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

# mpirun -np 4 ./generator md5 4096 65536 cpu
# ./cuda md5 4096 262144 cuda

# mpirun -np 4 ./generator sha1 4096 65536 cpu
# ./cuda sha1 4096 262144 cuda

# mpirun -np 4 ./generator hmac 4096 65536 cpu
# ./cuda hmac 4096 262144 cuda

# Test 3 (Crack)
# mpirun -np 4 ./generator des 4096 262144 cpu
# ./sort 4 des_4096-262144_cpu
# ./verified des des_4096-262144_cpu 4096
# ./generator des rand
# ./crack des file des_4096-262144_cpu des.txt

# mpirun -np 4 ./generator md5 4096 262144 cpu
# ./sort 4 md5_4096-262144_cpu
# ./verified md5 md5_4096-262144_cpu 4096
# ./generator md5 rand
# ./crack md5 file des_4096-262144_cpu md5.txt

# mpirun -np 4 ./generator sha1 4096 262144 cpu
# ./sort 4 sha1_4096-262144_cpu
# ./verified sha1 sha1_4096-262144_cpu 4096
# ./generator sha1 rand
# ./crack sha1 file sha1_4096-262144_cpu sha1.txt

# mpirun -np 4 ./generator hmac 4096 262144 cpu
# ./sort 4 hmac_4096-262144_cpu
# ./verified hmac hmac_4096-262144_cpu 4096
# ./generator hmac rand
# ./crack hmac file hmac_4096-262144_cpu hmac.txt

# ./cuda des 4096 262144 cuda
# ./sort 1 des_4096-262144_cuda
# ./generator des rand
# ./crack des file des_4096-262144_cuda des.txt

# ./cuda md5 4096 262144 cuda
# ./sort 1 md5_4096-262144_cuda
# ./generator md5 rand
# ./crack md5 file md5_4096-262144_cuda md5.txt

# ./cuda sha1 4096 262144 cuda
# ./sort 1 sha1_4096-262144_cuda
# ./generator sha1 rand
# ./crack sha1 file sha1_4096-262144_cuda sha1.txt

# ./cuda hmac 4096 262144 cuda
# ./sort 1 hmac_4096-262144_cuda
# ./generator hmac rand
# ./crack hmac file sha1_4096-262144_cuda hmac.txt

# Test 4 (Initialization Time)
./generator des rand
./crackcuda des file 4096 des.txt

./generator md5 rand
./crackcuda md5 file 4096 md5.txt

./generator sha1 rand
./crackcuda sha1 file 4096 sha1.txt