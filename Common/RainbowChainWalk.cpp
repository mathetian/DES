// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowAlgorithm.h"
#include "RainbowChainWalk.h"

namespace rainbowcrack
{

/// uint64_t RainbowChainWalk::m_keySpaceTotal = (1ull << 63) - 1 + (1ull << 63);
uint64_t     RainbowChainWalk::m_keySpaceTotal = (1ull << 31) - 1;
uint64_t     RainbowChainWalk::m_chainLen;
uint64_t     RainbowChainWalk::m_chainCount;
HASHROUTINE  RainbowChainWalk::m_algorithm;

void RainbowChainWalk::SetChainInfo(uint64_t chainLen, uint64_t chainCount, const char *type)
{
    m_chainLen   = chainLen;
    m_chainCount = chainCount;
    if(strcmp(type, "des") == 0)
    {
        m_algorithm = HASH_DES;
        /// m_keySpaceTotal = (1ull << 43) - 2 - (1ull << 8) - (1ull << 16) - (1ull << 24) - (1ull << 32) - (1ull << 40);
        m_keySpaceTotal = (1ull << 34) - 2 - (1ull << 8) - (1ull << 16) - (1ull << 24);
    }
    else if(strcmp(type, "md5") == 0)  m_algorithm = HASH_MD5;
    else if(strcmp(type, "sha1") == 0) m_algorithm = HASH_SHA1;
    else if(strcmp(type, "hmac") == 0) m_algorithm = HASH_HMAC;
    else assert(type && 0);
}

uint64_t RainbowChainWalk::GetRandomKey()
{
    RAND_bytes((unsigned char*)&m_key, 8);
    m_key = m_key & m_keySpaceTotal;
    return m_key;
}

void RainbowChainWalk::KeyToCipher()
{
    m_key = Crypt(m_key);
}

void RainbowChainWalk::KeyReduction(int nPos)
{
    m_key &= m_keySpaceTotal;
    if(nPos >= 1300)
    {
        m_key = (m_key + nPos) & m_keySpaceTotal;
        m_key = (m_key + (nPos << 8)) & m_keySpaceTotal;
        m_key = (m_key + ((nPos << 8) << 8)) & m_keySpaceTotal;
    }
}

uint64_t RainbowChainWalk::GetKey()
{
    return m_key & m_keySpaceTotal;
}

void 	 RainbowChainWalk::SetKey(uint64_t key)
{
    m_key = key & m_keySpaceTotal;
}

uint64_t RainbowChainWalk::Crypt(uint64_t key)
{
    unsigned char out[8];
    m_algorithm((unsigned char*)&key, 8, out);
    uint64_t result = (*(uint64_t*)out);
    return result;
}

};
