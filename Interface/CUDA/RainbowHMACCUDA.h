// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _HMAC_CUDA_H
#define _HMAC_CUDA_H

#include "RainbowSHA1CUDA.h"

namespace rainbowcrack
{

__device__ void SHA1_HMAC_Init(SHA1_CTX *ctx, const uint8_t *key, size_t keylen)
{
    size_t i;

    memset( ctx->ipad, 0x36, 64 );
    memset( ctx->opad, 0x5C, 64 );

    for( i = 0; i < keylen; i++ )
    {
        ctx->ipad[i] = (uint8_t)( ctx->ipad[i] ^ key[i] );
        ctx->opad[i] = (uint8_t)( ctx->opad[i] ^ key[i] );
    }

    SHA1_Init(ctx);
    SHA1_Update(ctx, ctx->ipad, 64);
}

__device__ void SHA1_HMAC_Update(SHA1_CTX *ctx, const uint8_t *input, size_t ilen )
{
    SHA1_Update(ctx, input, ilen);
}

__device__ void SHA1_HMAC_Final(SHA1_CTX *ctx, uint8_t *output)
{
    uint8_t tmpbuf[20];

    SHA1_Final( ctx, tmpbuf );
    SHA1_Init( ctx );
    SHA1_Update( ctx, ctx->opad, 64 );
    SHA1_Update( ctx, tmpbuf, 20 );
    SHA1_Final( ctx, output );
}

__device__ void SHA1_HMAC(const uint8_t *key, size_t keylen, const uint8_t *input, size_t ilen, uint8_t *output)
{
    SHA1_CTX ctx;

    SHA1_HMAC_Init( &ctx, key, keylen );
    SHA1_HMAC_Update( &ctx, input, ilen );
    SHA1_HMAC_Final( &ctx, output );
}

__device__ uint64_t Key2Ciper_HMAC(uint64_t key)
{
    ///uint8_t data[] = {'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'};
    uint8_t data[] = {'h'};
    uint8_t result[20], result_2[8];
    U64_2_CHAR(key, result_2);
    SHA1_HMAC(result_2, 8, data, 1, result);
    memcpy(result_2, result, 8);
    CHAR_2_U64(key, result_2);

    return key;
}

__device__ uint64_t Cipher2Key_HMAC(uint64_t key, int nPos)
{
    key &= totalSpace;
    if(nPos >= 1300)
    {
        key = (key + nPos) & totalSpace;
        key = (key + (nPos << 8)) & totalSpace;
        key = (key + ((nPos << 8) << 8)) & totalSpace;
    }

    return key;
}

__global__ void HMACCUDA(uint64_t *data)
{
    __syncthreads();

    uint64_t key = data[TX];
    for(int nPos = 0; nPos < CHAINLEN; nPos++)
        key = Cipher2Key_HMAC(Key2Ciper_HMAC(key), nPos);
    data[TX] = key;

    __syncthreads();
}

__global__ void HMACCUDA_ONCE(uint64_t *data)
{
    __syncthreads();

    uint64_t key = data[TX];
    data[TX] = Key2Ciper_HMAC(key);

    __syncthreads();
}


// __global__ void  HMACCrackCUDA(uint64_t *data)
// {
//     __syncthreads();

//     uint64_t ix  = TX, key = data[ix];
//     for(int nPos = (ix % 4096) + 1; nPos < 4096; nPos++)
//         key = Cipher2Key_HMAC(Key2Ciper_HMAC(key), nPos);
//     data[ix] = key;

//     ix = (1 << 19) - 1 - TX; key = data[ix];
//     for(int nPos = (ix % 4096) + 1; nPos < 4096; nPos++)
//         key = Cipher2Key_HMAC(Key2Ciper_HMAC(key), nPos);
//     data[ix] = key;

//     __syncthreads();
// }

};

#endif