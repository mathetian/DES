DESCrack
--------------------------------------------------

TEST 1, 
**************************************************
KeySpace, 2^20 (2^10 * 2^11)

Solve it very soon.

Successful Probability, 90% +

TEST 2, 
**************************************************
KeySpace, 2^28 (2^10 * 2^18)

Solve it very soon.

Successful Probability, 50% + (Not very sure and too tired to conduct experiments.)

TEST 3, 
**************************************************
KeySpace, 2^32 (2^11 * 2^21)

Precomputation Time,  7s for 2^11 * 2^13. So 2^21 t/s. Total time 20min.

Chain Reconstruction,  12s for 100 test cases. Disk time costs just 5% ~ 10%. 

Successful Probability, 72%.

----------------------------------------------------
Todo List,
----------------------------------------------------
1. CPU Parallel and GPU CUDA version

2. Distributed Crack tool

----------------------------------------------------
Windows version has been checked in.

Passed 20/31(32) bit tests, successful probability is very high. However, I found when I use parallel, the single process performance will becom bad
----------------------------------------------------
2014/01/24
Configuration, I have finished the testify of des_cuda. 

Currently, I found the performance is not very good. It needs 256 620M to generate all chains in 90 days. In another words, each core just equals to 12 cores. Sadly.

How to compute it,

In each second, it can compute 2^25.2 and 90 days have 2^22.8 seconds. So just need 2^56/2^(25.2+22.8)=2^8 = 256 GPU.

Assume we use CPU, it can compute 2^21.5 in i5 3317. So need 2^56/2^(21.5+22.8)=2^11.7=3326 core.(We must notice in parallel atomosphere, the performance is not four times than the single core.)
-------------------------------------------------------------------------------------------
Online Phase QA,
1. How to compute the all chains. 

As I can computed, if you assume chain len is equal to 2^20, it need to compute 2^39 times. As I have computed in CPU, it will need 51 hours in single core. In GPU, it need 4 hours.

2. How to use GPU in this phase.

Whether we need to compute all chains once time in GPU and How to use GPU to compute them as different length of chain will cost different time. I `think` GPU can just excute tasks and can't excute as CPU.

4. How to detect collision.

In current reference, I don't find clearly how to detect collision. In one paper, the author tells me first compute all chains, and then read files block to detect collision. I guess it's very right, however, GPU in this progress? Guess not as I don't find any materials to search collision in GPU. I also think GPU can't do these tasks as limited memory and heavy schedule overload.

We need serval experiments to verify or reject my previous result.

1. The time of search in CPU in single core or multiple core. And the search matrials is in global.

2. The limitation memory of GPU(and the limitation of `read` each time)
As we can use mmap or read large blocks of files, can they be put into GPU(or passed into gpu)

3. The method of use GPU to find collision.

4. Further experiments(like find the problem of core/thread and optimization of CUDA<which can be asked in stackoverflow><like different parameter, different prefix>)

----------------------------------------
Todo bugs,
1. recompute chain number(linux parallel)
2. chainlen(as computed much chains than needed)(cuda) 

-----------------------------------------
In i7, spend time 1.3s/2^20(random + write)
read, 0.17s/2^25
sort, 7s/2^25
write, 4s/2^25. (not cache time!sad).

Thus, total time 12s/2^25.
-----------------------------------------
Generater 2^25, spend time 21s?(45s? for 2^25*2(start & end))
For 2^25*2^27 binary search(maxinum, can have huge optimization), spend time 45s.(4k collision for 2^25/2^27/2^40.).

Need external sort to get better result.
External sort time, 27s/2^27 chains.
-------------------------------------------
So, 2^20*2^36 = 45s/32 * 2^9 = 2^10s(20min), collision.
Read, 2^11*0.17 = 2^10s(20min).

sort, 7*2^11 + 2^9*27 = 2^15...
-------------------------------------------
In i7 2600, each core 2^21.6.
Using hyper-threading, each core(two logic cores) 2^21.8

Same with E5 1620, maybe even better.

-------------------------------------------
Todo list:
1. k: 0.3 -> 1.0 performance change
1.0: 55%     40%
0.3: 2306860*1126 25%  15%

2. 0.87t (1782)  performance change
