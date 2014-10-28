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
