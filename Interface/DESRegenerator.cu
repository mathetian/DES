// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "DESCuda.h"
#include "DESCipherSet.h"
using namespace descrack;

__device__ int GenerateKey(uint64_t key, uint64_t * store)
{
    uint32_t c, d, t, s, t2;
    uint64_t tmp;
    /**c: low 32 bits, d high 32 bits**/
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

    //one round, 0.25s*16=4s

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

/**
** DESGeneratorCUDA, the really entrance function
** Rainbow
**/
__global__ void  DESGeneratorCUDA(uint64_t * data)
{
    for(int i=0; i<256; i++)
    {
        ((uint64_t *)des_SP)[i] = ((uint64_t *)des_d_sp_c)[i];
    }

    /*((uint64_t *)des_SP)[threadIdx.x] = ((uint64_t *)des_d_sp_c)[threadIdx.x];*/

    __syncthreads();

    register uint64_t m_nIndex = data[TX];
    uint64_t roundKeys[16];

    uint64_t needCal = CHAINLEN - TX - 1;

    for(int nPos = 0; nPos < needCal; nPos++)
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

    data[TX] = m_nIndex;

    __syncthreads();
}

void Usage()
{
    Logo();
    printf("Usage: regenerate text chainLen chainCount encryptedText \n");
    printf("                  file chainLen chainCount encryptedFile \n\n");

    printf("example 1: regenerate text chainLen chainCount 12345 7831224 541234 3827427\n");
    printf("example 2: regenerate file chainLen chainCount fileName\n\n");
}

void Regenerator(DESCipherSet *p_cs, uint64_t chainLen, uint64_t chainCount)
{
    assert(chainLen <= ALL*sizeof(uint64_t));

    uint64_t starts[ALL], ends[ALL];

    uint64_t *cudaIn;
    int size = chainLen*sizeof(uint64_t);
    _CUDA(cudaMalloc((void**)&cudaIn , size));

    while(p_cs -> AnyKeyLeft() == true)
    {
        uint64_t key = p_cs -> GetLeftKey();
        p_cs -> Done(0);

        stringstream ss;
        ss << key << ".txt";

        /// Won't check duplicate files
        FILE *file = fopen(ss.str().c_str(), "ab+");

        assert(file);

        for(uint64_t nPos = 0; nPos < chainLen ; nPos++)
        {
            starts[nPos] = (key & totalSpaceT);
            int nnpos = nPos;

            if(nPos < 1300) nnpos = 0;
            starts[nPos] = (starts[nPos] + nnpos) & totalSpaceT;
            starts[nPos] = (starts[nPos] + (nnpos << 8)) & totalSpaceT;
            starts[nPos] = (starts[nPos] + ((nnpos << 8) << 8)) & totalSpaceT;
        }

        _CUDA(cudaMemcpy(cudaIn, starts, size,cudaMemcpyHostToDevice));

        DESGeneratorCUDA<<<BLOCK_LENGTH, MAX_THREAD>>>(cudaIn);

        _CUDA(cudaMemcpy(ends, cudaIn, size, cudaMemcpyDeviceToHost));

        for(uint64_t pos = 0; pos < chainLen ; pos++)
        {
            int flag1 = fwrite((char*)&(starts[pos]), sizeof(uint64_t), 1, file);
            int flag2 = fwrite((char*)&(ends[pos]), sizeof(uint64_t), 1, file);
            assert((flag1 == 1) && (flag2 == 1));
        }

        fclose(file);
    }
}

int main(int argc, char *argv[])
{
    int keyNum, index;
    uint64_t chainLen, chainCount;
    DESCipherSet  * p_cs = DESCipherSet::GetInstance();

    if(argc <= 3)
    {
        Usage();
        return 0;
    }
    if(strcmp(argv[1],"file") == 0)
    {
        if(argc != 5)
        {
            Usage();
            return 0;
        }
        chainLen   = atoll(argv[2]);
        chainCount = atoll(argv[3]);

        FILE * file = fopen(argv[4],"rb");
        assert(file && "main fopen error\n");

        /// To verify the result of crack, we put the plain password here
        RainbowChain chain;
        while(fread((char*)&chain, sizeof(RainbowChain), 1, file))
            p_cs -> AddKey(chain.nEndKey);

        fclose(file);
    }
    else if(strcmp(argv[1],"text") == 0)
    {
        chainLen   = atoll(argv[2]);
        chainCount = atoll(argv[3]);

        keyNum = argc - 4;
        for(index = 0; index < keyNum; index++)
            p_cs -> AddKey(atoll(argv[index + 4]));
    }
    else
    {
        Usage();
        return 0;
    }

    Regenerator(p_cs, chainLen, chainCount);

    return 0;
}