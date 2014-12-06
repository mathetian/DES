// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _HMAC_CUDA_H
#define _HMAC_CUDA_H

namespace rainbowcrack
{

#define F(x, y, z)          ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)          ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)          (((x) ^ (y)) ^ (z))
#define H2(x, y, z)         ((x) ^ ((y) ^ (z)))
#define I(x, y, z)          ((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

#define SET(n) \
    (block[(n)] = \
    (uint32_t)ptr[(n) * 4] | \
    ((uint32_t)ptr[(n) * 4 + 1] << 8) | \
    ((uint32_t)ptr[(n) * 4 + 2] << 16) | \
    ((uint32_t)ptr[(n) * 4 + 3] << 24))
#define GET(n) \
    (block[(n)])

__device__ void BODY(register uint32_t &a, register uint32_t &b, 
    register uint32_t &c, register uint32_t &d, uint8_t *ptr)
{
    register uint32_t saved_a, saved_b, saved_c, saved_d;
    uint32_t block[16];

    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;   
}

__device__ void HMAC_MD5(uint8_t *key, size_t keylen, uint8_t *input, uint8_t *output)
{
    register uint32_t lo, hi, a, b, c, d, i; uint8_t buffer[64]; 

    {
        lo = hi = 0; 
        a = 0x67452301; b = 0xefcdab89;
        c = 0x98badcfe; d = 0x10325476; 

        {
            memset(buffer, 0x36, 64 );
            for( i = 0; i < keylen; i++ )
                buffer[i] = (uint8_t)(buffer[i] ^ key[i] );
            BODY(a, b, c, d, buffer); 
        }
        
        {
            buffer[0] = input[0]; buffer[1] = 0x80;
            memset(&buffer[2], 0, 64 - 2 - 8);
            lo = 1; lo <<= 3;
            buffer[56] = lo;
            buffer[57] = lo >> 8;
            buffer[58] = lo >> 16;
            buffer[59] = lo >> 24;
            buffer[60] = hi;
            buffer[61] = hi >> 8;
            buffer[62] = hi >> 16;
            buffer[63] = hi >> 24;
            BODY(a, b, c, d, buffer);
        } 

        output[0] = a;
        output[1] = a >> 8;
        output[2] = a >> 16;
        output[3] = a >> 24;
        output[4] = b;
        output[5] = b >> 8;
        output[6] = b >> 16;
        output[7] = b >> 24;
        output[8] = c;
        output[9] = c >> 8;
        output[10] = c >> 16;
        output[11] = c >> 24;
        output[12] = d;
        output[13] = d >> 8;
        output[14] = d >> 16;
        output[15] = d >> 24;
    }
    
    {
        lo = hi = 0; 
        a = 0x67452301; b = 0xefcdab89;
        c = 0x98badcfe; d = 0x10325476; 

        {
            memset(buffer, 0x5C, 64 );
            for( i = 0; i < keylen; i++ )
                buffer[i] = (uint8_t)(buffer[i] ^ key[i] );
            BODY(a, b, c, d, buffer); 
        }
        
        {
            memcpy(buffer, output, 16);
            memset(&buffer[16], 0, 64 - 16 - 8);
            lo = 16; lo <<= 3;
            buffer[56] = lo;
            buffer[57] = lo >> 8;
            buffer[58] = lo >> 16;
            buffer[59] = lo >> 24;
            buffer[60] = hi;
            buffer[61] = hi >> 8;
            buffer[62] = hi >> 16;
            buffer[63] = hi >> 24;
            BODY(a, b, c, d, buffer);
        }
        
        output[0] = a;
        output[1] = a >> 8;
        output[2] = a >> 16;
        output[3] = a >> 24;
        output[4] = b;
        output[5] = b >> 8;
        output[6] = b >> 16;
        output[7] = b >> 24;
        output[8] = c;
        output[9] = c >> 8;
        output[10] = c >> 16;
        output[11] = c >> 24;
        output[12] = d;
        output[13] = d >> 8;
        output[14] = d >> 16;
        output[15] = d >> 24;
    }
}

__device__ uint64_t Key2Ciper_HMAC_MD5(uint64_t key)
{
    register uint8_t data[] = {'h'}, result[16];

    U64_2_CHAR(key, result);
    HMAC_MD5(result, 8, data, result);
    CHAR_2_U64(key, result);

    return key;
}

__global__ void HMAC_MD5_CUDA(uint64_t *data)
{
    register uint64_t key = data[TX];
    for(int nPos = 0; nPos < CHAINLEN; nPos++)
        key = Cipher2Key(Key2Ciper_HMAC_MD5(key), nPos);
    data[TX] = key;
}

__global__ void HMAC_MD5_CrackCUDA(uint64_t *data)
{
}


};

#endif