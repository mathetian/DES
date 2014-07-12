// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "DESCipherSet.h"

namespace descrack
{

DESCipherSet *DESCipherSet::p_cs = NULL;

DESCipherSet * DESCipherSet::GetInstance()
{
    if(p_cs == NULL) p_cs = new DESCipherSet();

    return p_cs;
}

DESCipherSet::DESCipherSet() : index(0), solve(false)
{
}

void DESCipherSet::AddKey(uint64_t cipherKey)
{
    m_vKeys.push_back(cipherKey);
}

bool DESCipherSet::AnyKeyLeft()
{
    return index == m_vKeys.size() ? false : true;
}

uint64_t DESCipherSet::GetLeftKey()
{
    return m_vKeys.at(index);
}

void DESCipherSet::AddResult(uint64_t cipherKey, uint64_t key)
{
    solve = true;

    m_maps[cipherKey].push_back(key);
}

void DESCipherSet::Done(uint64_t cipherKey)
{
    solve = false;
    index++;
}

bool DESCipherSet::Solved()
{
    return solve;
}

int DESCipherSet::GetKeyFoundNum()
{
    return m_vFound.size();
}

int DESCipherSet::Detect(RainbowChain chain)
{
    vector<uint64_t> tmp = m_maps[chain.nEndKey];

    if(tmp.size() == 0)
        return false;

    cout<< chain.nEndKey << " " << tmp.size() <<endl;

    for(uint64_t i = 0; i < tmp.size(); i++)
    {
        if(tmp.at(i) == chain.nStartKey)
            return 1;
    }

    return 0;
}

int DESCipherSet::GetRemainCount()
{
    return m_vKeys.size() - index;
}

};