RainbowCrack
=====

RainbowCrack is an implementation of Rainbow Table.

Benchmark:

Compared. 
====
DES/MD5/SHA1/HMAC

Test1. Encryption Compared with `OpenSSL`

Test2. Rainbow in CPU Compard WITH CUDA

PS 1: HMAC too slow & HMAC in CUDA is wrong

Speed
=====
DES CPU: 2^20.64 per second (log(10000*4096/25.0, 2)) (one core)
DES GPU: 2^25.35 per second (log((2**30)/25.0, 2)) (one gpu)

MD5 CPU: 2^21.38 per second (log(10000*4096/15.0, 2)) (one core)
MD5 GPU: 2^25.35 per second (log((2**30)/25.0, 2)) (one gpu)

SHA1 CPU: 2^20.96 per second (log(10000*4096/20.0, 2)) (one core)
SHA1 GPU: 2^23.79 per second (log((2**30)/74.0, 2)) (one gpu)

HMAC CPU: 2^18.5 per second (log(10000*4096/110.0, 2)) (one core)
HMAC GPU: 2^21.5 per second (log((2**30)/360.0, 2)) (one gpu)

HPC:
DES CPU: 2^21.03 per second (log(10000*4096/19.0, 2)) (one core)
DES GPU: 2^28.41 per second (log((2**30)/3.0, 2)) (one gpu)

MD5 CPU: 2^21.48 per second (log(10000*4096/14.0, 2)) (one core)
MD5 GPU: 2^28.41 per second (log((2**30)/3.0, 2)) (one gpu)

SHA1 CPU: 2^21.38 per second (log(10000*4096/15.0, 2)) (one core)
SHA1 GPU: 2^27.19 per second (log((2**30)/7.0, 2)) (one gpu)

HMAC CPU: 2^19.43 per second (log(10000*4096/58.0, 2)) (one core)
HMAC GPU: 2^25.09 per second (log((2**30)/30.0, 2)) (one gpu)

Crack
=====
DES (2**30)
-------------------------------------------------------
Statistics
-------------------------------------------------------
Key found             : 67
Total time            : 488 s, 187429 us
Total init time       : 296 s, 237840 us
Total disk access time: 0 s, 97907 us
Total compare time    : 191 s, 851682 us
Total chains steps    : 26214400
Total false alarms    : 410400
Detected 35 numbers

MD5 (2**30)
-------------------------------------------------------
Statistics
-------------------------------------------------------
Key found             : 61
Total time            : 221 s, 190025 us
Total init time       : 164 s, 419520 us
Total disk access time: 0 s, 106740 us
Total compare time    : 56 s, 663765 us
Total chains steps    : 26214400
Total false alarms    : 205766
Detected 48 numbers

SHA1 (2**30)
-------------------------------------------------------
Statistics
-------------------------------------------------------
Key found             : 65
Total time            : 234 s, 557858 us
Total init time       : 174 s, 44006 us
Total disk access time: 0 s, 107614 us
Total compare time    : 60 s, 406238 us
Total chains steps    : 26214400
Total false alarms    : 205976
Detected 41 numbers


HMAC (2**30)
-------------------------------------------------------
Statistics
-------------------------------------------------------
Key found             : 61
Total time            : 1310 s, 325729 us
Total init time       : 978 s, 352615 us
Total disk access time: 0 s, 96495 us
Total compare time    : 331 s, 876619 us
Total chains steps    : 26214400
Total false alarms    : 206126
Detected 44 numbers

Initialization Time
===
DES:  2^25.26 per second (log((2**30)/26.6, 2)) (one gpu)

MD5:  2^25.36 per second (log((2**30)/24.8, 2)) (one gpu)

SHA1: 2^23.78 per second (log((2**30)/74.2, 2)) (one gpu)

Todo List
===
1. Review HMAC(Speed and Others)

2. RainbowCrackCUDA(Initialization -> CUDA)

PS 1: uint8_t cost too much time than void*