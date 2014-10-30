// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowCipherSet.h"

namespace rainbowcrack
{

RainbowCipherSet *RainbowCipherSet::p_cs = NULL;

RainbowCipherSet * RainbowCipherSet::GetInstance()
{
    if(p_cs == NULL) p_cs = new RainbowCipherSet();

    return p_cs;
}

RainbowCipherSet::RainbowCipherSet() : index(0), solve(false)
{
}

void RainbowCipherSet::AddKey(uint64_t cipherKey)
{
    m_vKeys.push_back(cipherKey);
}

bool RainbowCipherSet::Finished()
{
    return index == m_vKeys.size() ? false : true;
}

uint64_t RainbowCipherSet::GetLastKey()
{
    return m_vKeys[index];
}

void RainbowCipherSet::AddResult(uint64_t cipherKey, uint64_t key)
{
    solve = true;
    m_maps[cipherKey].push_back(key);
}

void RainbowCipherSet::Done()
{
    solve = false;
    index++;
}

bool RainbowCipherSet::Solved()
{
    return solve;
}

int RainbowCipherSet::GetKeyFoundNum()
{
    return m_maps.size();
}

int RainbowCipherSet::Detect(RainbowChain chain)
{
    if(m_maps[chain.nEndKey].size() == 0) return false;

    for(uint64_t i = 0; i < m_maps[chain.nEndKey].size(); i++)
        if(m_maps[chain.nEndKey][i] == chain.nStartKey) return 1;

    return 0;
}

int RainbowCipherSet::GetRemainCount()
{
    return m_vKeys.size() - index;
}

};