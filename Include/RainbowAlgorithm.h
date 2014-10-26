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

#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/x509.h>
#include <openssl/ssl.h>

void HASH_DES(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    static unsigned char m_dplainText[8] = {0x6D,0x6F,0x29,0x5A,0x30,0x55,0x32,0x28};

    des_key_schedule ks;
    des_cblock key_56;
    memcpy(key_56, pPlain, nPlainLen);
    DES_set_key_unchecked(&key_56, &ks);
    unsigned char out[8];
    des_ecb_encrypt(&m_dplainText, &out, ks, DES_ENCRYPT);
    memcpy(pHash, out, 8);
}

void HASH_MD5(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    unsigned char out[16];
    MD5(pPlain, nPlainLen, out);
    memcpy(pHash, out, 8);
}

void HASH_SHA1(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
    unsigned char out[20];
    SHA1(pPlain, nPlainLen, out);
    memcpy(pHash, out, 8);
}

void HASH_HMAC(unsigned char *pPlain, int nPlainLen, unsigned char* pHash)
{
    unsigned char out[20]; unsigned int result_len; uint8_t data[] = {'h'};
    HMAC(EVP_sha1(), pPlain, nPlainLen, data, 1, out, &result_len);
    memcpy(pHash, out, 8);
}

#endif