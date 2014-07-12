// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "DESChainWalkContext.h"

namespace descrack
{

uint64_t DESChainWalkContext::m_plainText     = 0x305532286D6F295A;
unsigned char DESChainWalkContext::m_dplainText[8] = {0x6D,0x6F,0x29,0x5A,0x30,0x55,0x32,0x28};

uint64_t DESChainWalkContext::m_keySpaceTotal = (1ull << 43) - 2 - (1ull << 8) - (1ull << 16) - (1ull << 24) - (1ull << 32) - (1ull << 40);

uint64_t DESChainWalkContext::m_chainLen;
uint64_t DESChainWalkContext::m_chainCount;

void DESChainWalkContext::SetChainInfo(uint64_t chainLen, uint64_t chainCount)
{
    m_chainLen   = chainLen;
    m_chainCount = chainCount;
}

uint64_t DESChainWalkContext::GetRandomKey()
{
    RAND_bytes((unsigned char*)&m_nIndex,8);
    m_nIndex = m_nIndex & m_keySpaceTotal;
    return m_nIndex;
}

void DESChainWalkContext::KeyToCipher()
{
    des_key_schedule ks;
    unsigned char out[8];
    SetupDESKey(m_nIndex,ks);
    memset(out,0,8);
    des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);
    CipherToKey(out);
}

void DESChainWalkContext::CipherToKey(unsigned char * out)
{
    Arr7ToU56(out, m_nIndex);
    m_nIndex &= m_keySpaceTotal;
}

void DESChainWalkContext::KeyReduction(int nPos)
{
    if(nPos < 1300) nPos = 0;
    m_nIndex = (m_nIndex + nPos) & m_keySpaceTotal;
    m_nIndex = (m_nIndex + (nPos << 8)) & m_keySpaceTotal;
    m_nIndex = (m_nIndex + ((nPos << 8) << 8)) & m_keySpaceTotal;
}

uint64_t DESChainWalkContext::GetKey()
{
    return m_nIndex & m_keySpaceTotal;
}

void 	 DESChainWalkContext::SetKey(uint64_t key)
{
    m_nIndex = key & m_keySpaceTotal;
}

uint64_t DESChainWalkContext::Crypt(uint64_t key)
{
    unsigned char out[8];
    memset(out,0,8);
    des_key_schedule ks;
    SetupDESKey(key, ks);

    des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);

    Arr7ToU56(out, key);
    key &= m_keySpaceTotal;

    return key;
}

};