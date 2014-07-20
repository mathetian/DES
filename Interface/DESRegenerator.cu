// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "DESCuda.h"
#include "DESCipherSet.h"
using namespace descrack;

void Usage()
{
    Logo();
    printf("Usage: regenerate file chainLen encryptedFile \n\n");

    printf("example 1: regenerate file chainLen encryptedFile\n\n");
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
    RoundKey0(1) ;
    RoundKey1(2) ;
    RoundKey1(3);
    RoundKey1(4);
    RoundKey1(5) ;
    RoundKey1(6) ;
    RoundKey1(7);
    RoundKey0(8);
    RoundKey1(9) ;
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

    rs=(((uint64_t)right)<<32)|left; //why, who can explain it
    return rs;
}

__global__ void  DESGeneratorCUDA(uint64_t *data)
{
    for(int i=0; i<256; i++)
        ((uint64_t *)des_SP)[i] = ((uint64_t *)des_d_sp_c)[i];

    __syncthreads();

    uint64_t tx = TX / 4096;
    uint64_t st = tx*4096*2;
    uint64_t ix = st + (TX % 4096);

    register uint64_t m_nIndex = data[ix];

    uint64_t roundKeys[16];

    for(int nPos = (TX % 4096) + 1; nPos < 4096; nPos++)
    {
        GenerateKey(m_nIndex, roundKeys);
        m_nIndex  = DESOneTime(roundKeys);
        m_nIndex &= totalSpace;

        int nnpos = nPos;
        if(nPos < 1300) nnpos = 0;
        m_nIndex = (m_nIndex + nnpos) & totalSpace;
        m_nIndex = (m_nIndex + (nnpos << 8)) & totalSpace;
        m_nIndex = (m_nIndex + ((nnpos << 8) << 8)) & totalSpace;
    }

    data[ix] = m_nIndex;

    ix = st + 2*4096 - TX % 4096 - 1;
    st = st + 4096;

    m_nIndex = data[ix];

    for(int nPos = 4096 - (TX % 4096); nPos < 4096; nPos++)
    {
        GenerateKey(m_nIndex, roundKeys);
        m_nIndex  = DESOneTime(roundKeys);
        m_nIndex &= totalSpace;

        int nnpos = nPos;
        if(nPos < 1300) nnpos = 0;
        m_nIndex = (m_nIndex + nnpos) & totalSpace;
        m_nIndex = (m_nIndex + (nnpos << 8)) & totalSpace;
        m_nIndex = (m_nIndex + ((nnpos << 8) << 8)) & totalSpace;
    }

    __syncthreads();
}

void Regenerator(DESCipherSet *p_cs, uint64_t chainLen)
{
    assert(chainLen == 4096);

    uint64_t starts[ALL*2], ends[ALL*2];

    uint64_t *cudaIn;
    int size = ALL*sizeof(uint64_t)*2;
    _CUDA(cudaMalloc((void**)&cudaIn , size));

    FILE *result = fopen("Result.txt", "wb+");
    assert(result);

    while(p_cs -> GetRemainCount() < 128)
    {
        uint64_t keys[128];

        for(int i = 0; i < 128; i++)
        {
            keys[i] = p_cs -> GetLeftKey();
            p_cs -> Done(0);
        }

        for(int i = 0; i < 128; i++)
        {
            for(uint64_t nPos = 0; nPos < chainLen ; nPos++)
            {
                starts[i*chainLen + nPos] = (keys[i] & totalSpaceT);
                int nnpos = nPos;

                if(nPos < 1300) nnpos = 0;
                starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + nnpos) & totalSpaceT;
                starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + (nnpos << 8)) & totalSpaceT;
                starts[i*chainLen + nPos] = (starts[i*chainLen + nPos] + ((nnpos << 8) << 8)) & totalSpaceT;
            }
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size, cudaMemcpyHostToDevice));

        DESGeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(int i = 0; i < 128; i++)
        {
            for(uint64_t pos = 0; pos < chainLen ; pos++)
            {
                int flag1 = fwrite((char*)&(ends[i*chainLen + pos]), sizeof(uint64_t), 1, result);
                assert(flag1 == 1);
            }
        }
    }

    fclose(result);
}

int main(int argc, char *argv[])
{
    uint64_t chainLen;
    DESCipherSet  *p_cs = DESCipherSet::GetInstance();

    if(argc != 4 || strcmp(argv[1],"file") != 0)
    {
        Usage();
        return 0;
    }

    FILE *file = fopen(argv[3], "rb");
    assert(file);

    RainbowChain chain;

    fclose(file);

    while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
        p_cs -> AddKey(chain.nEndKey);

    chainLen = atoll(argv[2]);

    Regenerator(p_cs, chainLen);

    return 0;
}