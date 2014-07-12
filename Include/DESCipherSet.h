// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _CIPHER_SET_H
#define _CIPHER_SET_H

#include "Common.h"
using namespace utils;

namespace descrack
{

class DESCipherSet
{

public:
    static DESCipherSet * GetInstance();

public:
    void     AddKey(uint64_t cipherKey);
    bool     AnyKeyLeft();
    uint64_t GetLeftKey();

    void     AddResult(uint64_t cipherKey,uint64_t key);
    void     Done(uint64_t cipherKey);
    bool     Solved();
    int      GetKeyFoundNum();
    int      Detect(RainbowChain chain);

    int      GetRemainCount();

private:
    DESCipherSet();
    static DESCipherSet * p_cs;

private:
    vector<uint64_t> m_vKeys;
    vector<pair<uint64_t,vector<uint64_t> > > m_vFound;
    map<uint64_t, vector<uint64_t> > m_maps;
    uint32_t index;
    bool solve;
};

};

#endif