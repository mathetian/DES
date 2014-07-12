// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "TimeStamp.h"
using namespace utils;

#include "DESCuda.h"
using namespace descrack;

void Usage()
{
    Logo();
    printf("Usage: gencuda   chainLen chainCount suffix\n\n");

    printf("example 1: gencuda 1000 10000 suffix\n");
}

__device__ int GenerateKey(uint64_t key, uint64_t * store)
{
    uint32_t c, d, t, s, t2;
    uint64_t tmp;
    c = ((1ull << 32) - 1) & key;
    d = (key >> 32);

    PERM_OP (d,c,t,4,0x0f0f0f0fL);
    HPERM_OP(c,t, -2,0xcccc0000L);
    HPERM_OP(d,t, -2,0xcccc0000L);
    PERM_OP (d,c,t,1,0x55555555L);
    PERM_OP (c,d,t,8,0x00ff00ffL);
    PERM_OP (d,c,t,1,0x55555555L);

    d =	(((d&0x000000ffL)<<16L)| (d&0x0000ff00L)     |
         ((d&0x00ff0000L)>>16L)|((c&0xf0000000L)>>4L));
    c&=0x0fffffffL;


    RoundKey0(0);
    RoundKey0(1);
    RoundKey1(2);
    RoundKey1(3);
    RoundKey1(4);
    RoundKey1(5);
    RoundKey1(6);
    RoundKey1(7);
    RoundKey0(8);
    RoundKey1(9);
    RoundKey1(10);
    RoundKey1(11);
    RoundKey1(12);
    RoundKey1(13);
    RoundKey1(14);
    RoundKey0(15);

    return 0;
}

__device__ uint64_t DESOneTime(uint64_t * roundKeys)
{
    uint64_t rs;
    uint32_t right = plRight, left = plLeft;

    IP(right, left);

    left  = ROTATE(left,29)&0xffffffffL;
    right = ROTATE(right,29)&0xffffffffL;

    D_ENCRYPT(left,right, 0);
    D_ENCRYPT(right,left, 1);
    D_ENCRYPT(left,right, 2);
    D_ENCRYPT(right,left, 3);
    D_ENCRYPT(left,right, 4);
    D_ENCRYPT(right,left, 5);
    D_ENCRYPT(left,right, 6);
    D_ENCRYPT(right,left, 7);
    D_ENCRYPT(left,right, 8);
    D_ENCRYPT(right,left, 9);
    D_ENCRYPT(left,right,10);
    D_ENCRYPT(right,left,11);
    D_ENCRYPT(left,right,12);
    D_ENCRYPT(right,left,13);
    D_ENCRYPT(left,right,14);
    D_ENCRYPT(right,left,15);

    left  = ROTATE(left,3)&0xffffffffL;
    right = ROTATE(right,3)&0xffffffffL;

    FP(right, left);

    rs=(((uint64_t)right)<<32) | left;

    return rs;
}

__global__ void  DESGeneratorCUDA(uint64_t * data)
{
    for(int i=0; i < 256; i++)
        ((uint64_t *)des_SP)[i] = ((uint64_t *)des_d_sp_c)[i];

    __syncthreads();

    register uint64_t m_nIndex = data[TX];
    uint64_t roundKeys[16];

    for(int nPos = 0; nPos < CHAINLEN; nPos++)
    {
        GenerateKey(m_nIndex,roundKeys);

        m_nIndex  = DESOneTime(roundKeys);
        m_nIndex &= totalSpace;

        int nnpos = nPos;

        if(nPos < 1300)
            nnpos = 0;

        m_nIndex = (m_nIndex + nnpos) & totalSpace;
        m_nIndex = (m_nIndex + (nnpos << 8)) & totalSpace;
        m_nIndex = (m_nIndex + ((nnpos << 8) << 8)) & totalSpace;
    }

    data[TX] = m_nIndex;

    __syncthreads();
}

uint64_t Convert(uint64_t num, int time)
{
    assert(num < 8);

    uint64_t rs = 0, tmp = 0;

    for(int i = 0; i < time; i++)
    {
        tmp = num & ((1ull << 7) - 1);
        tmp <<= 1;
        tmp <<= (8*i);
        rs |= tmp;
        num >>= 7;
    }

    return rs;
}

void DESGenerator(uint64_t chainLen, uint64_t chainCount, const char * suffix)
{
    char fileName[100];
    memset(fileName, 0, 100);

    sprintf(fileName,"DES_%lld-%lld_%s-cuda", (long long)chainLen, (long long)chainCount,suffix);

    FILE * file = fopen(fileName, "ab+");
    assert(file);

    uint64_t nDatalen = GetFileLen(file);

    assert((nDatalen & ((1 << 4) - 1)) == 0);

    int remainCount =  chainCount - (nDatalen >> 4);

    int time1 = (remainCount + ALL - 1)/ALL;

    uint64_t size = sizeof(uint64_t)*ALL;

    uint64_t * cudaIn;
    uint64_t starts[ALL], ends[ALL];

    _CUDA(cudaMalloc((void**)&cudaIn , size));

    printf("Need to compute %d rounds %lld\n", time1, (long long)remainCount);

    for(int round = 0; round < time1; round++)
    {
        printf("Begin compute the %d round\n", round+1);

        TimeStamp tms;
        tms.StartTime();

        for(uint64_t i = 0; i < ALL; i++)
        {
            starts[i] = Convert(round*ALL + i, 6);
            starts[i] &= totalSpaceT;
        }

        _CUDA(cudaMemcpy(cudaIn,starts,size,cudaMemcpyHostToDevice));

        DESGeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends,cudaIn,size,cudaMemcpyDeviceToHost));

        for(uint64_t i = 0; i < ALL; i++)
        {
            int flag1 = fwrite((char*)&(starts[i]),sizeof(uint64_t),1,file);
            int flag2 = fwrite((char*)&(ends[i]),sizeof(uint64_t),1,file);
            assert((flag1 == 1) && (flag2 == 1));
        }

        printf("End compute the %d round\n", round+1);

        tms.StopTime("StopTime: ");
    }
}

int main(int argc, char * argv[])
{
    if(argc != 4)
    {
        Usage();
        return 1;
    }

    uint64_t chainLen, chainCount;
    char suffix[100];

    memset(suffix,0,sizeof(suffix));

    chainLen   = atoll(argv[1]);
    chainCount = atoll(argv[2]);
    memcpy(suffix,argv[3],strlen(argv[3]));

    DESGenerator(chainLen, chainCount, suffix);

    return 0;
}
