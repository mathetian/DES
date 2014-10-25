// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _CIPHER_SET_H
#define _CIPHER_SET_H

#include "Common.h"
using namespace utils;

namespace rainbowcrack
{

class RainbowCipherSet
{

public:
    static RainbowCipherSet * GetInstance();

public:
    void     AddKey(uint64_t cipherKey);
    bool     Finished();
    uint64_t GetLastKey();
    void     AddResult(uint64_t cipherKey,uint64_t key);
    void     Done();
    bool     Solved();
    int      GetKeyFoundNum();
    int      Detect(RainbowChain chain);
    int      GetRemainCount();

private:
    RainbowCipherSet();
    static RainbowCipherSet * p_cs;

private:
    vector<uint64_t> m_vKeys;
    vector<pair<uint64_t,vector<uint64_t> > > m_vFound;
    map<uint64_t, vector<uint64_t> > m_maps;
    uint32_t index;
    bool solve;
};

};

#endif