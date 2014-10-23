// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowAlgorithm.h"

#include "RainbowChainWalk.h"

namespace rainbowcrack
{

uint64_t RainbowChainWalk::m_keySpaceTotal = (1ull << 43) - 2 - (1ull << 8) - (1ull << 16) - (1ull << 24) - (1ull << 32) - (1ull << 40);

uint64_t     RainbowChainWalk::m_chainLen;
uint64_t     RainbowChainWalk::m_chainCount;
HASHROUTINE  RainbowChainWalk::m_algorithm;

void RainbowChainWalk::SetChainInfo(uint64_t chainLen, uint64_t chainCount, const char *type)
{
    m_chainLen   = chainLen;
    m_chainCount = chainCount;
    if(strcmp(type, "DES") == 0)
    {
        m_algorithm = DES;
        m_keySpaceTotal = (1ull << 43) - 2 - (1ull << 8) - (1ull << 16) - (1ull << 24) - (1ull << 32) - (1ull << 40);
    }
    else if(strcmp(type, "MD5") == 0)
    {
        m_algorithm = MD5;
    } 
}

uint64_t RainbowChainWalk::GetRandomKey()
{
    RAND_bytes((unsigned char*)&m_nIndex, 8);
    m_nIndex = m_nIndex & m_keySpaceTotal;
    return m_nIndex;
}

void RainbowChainWalk::KeyToCipher()
{
    m_nIndex = Crypt(m_nIndex);
}

void RainbowChainWalk::KeyReduction(int nPos)
{
    if(nPos < 1300) nPos = 0;
    m_nIndex = (m_nIndex + nPos) & m_keySpaceTotal;
    m_nIndex = (m_nIndex + (nPos << 8)) & m_keySpaceTotal;
    m_nIndex = (m_nIndex + ((nPos << 8) << 8)) & m_keySpaceTotal;
}

uint64_t RainbowChainWalk::GetKey()
{
    return m_nIndex & m_keySpaceTotal;
}

void 	 RainbowChainWalk::SetKey(uint64_t key)
{
    m_nIndex = key & m_keySpaceTotal;
}

uint64_t RainbowChainWalk::Crypt(uint64_t key)
{
    unsigned char out[8];
    m_algorithm((unsigned char*)&key, 8, out);
    key = (*(uint64_t*)out) & m_keySpaceTotal;

    return key;
}

};