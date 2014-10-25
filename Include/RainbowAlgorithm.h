// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _ALGORITHM_H
#define _ALGORITHM_H

#include "Common.h"
using namespace utils;

#include <openssl/des.h>
#include <openssl/md5.h>
#include <openssl/sha.h>

void DES(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    /// static uint64_t m_plainText          = 0x305532286D6F295A;
    static unsigned char m_dplainText[8] = {0x6D,0x6F,0x29,0x5A,0x30,0x55,0x32,0x28};

    des_key_schedule ks;
    des_cblock key_56;
    memcpy(key_56, pPlain, nPlainLen);
    DES_set_key_unchecked(&key_56, &ks);
    unsigned char out[8];
    des_ecb_encrypt(&m_dplainText, &out, ks, DES_ENCRYPT);
    memcpy(pHash, out, nPlainLen);
}

void MD5(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    MD5(pPlain, nPlainLen, pHash);
}

void SHA1(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    SHA1(pPlain, nPlainLen, pHash);
}

#endif