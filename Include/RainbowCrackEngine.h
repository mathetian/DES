// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _CRACK_ENGINE_H
#define _CRACK_ENGINE_H

#include "Common.h"
#include "MemoryPool.h"
using namespace utils;

#include "RainbowCipherSet.h"
#include "RainbowChainWalk.h"

namespace rainbowcrack
{

class RainbowCrackEngine
{
public:
    RainbowCrackEngine();

public:
    void  Run(const char *fileName, const char *type);

public:
    /// Statistics Public Function
    struct timeval   GetDiskTime();
    struct timeval   GetTotalTime();
    struct timeval   GetInitTime();
    struct timeval   GetCompareTime();
    uint64_t   		 GetTotalChains();
    uint64_t		 GetFalseAlarms();

private:
    uint64_t  BinarySearch(RainbowChain * pChain, uint64_t pChainCount, uint64_t nIndex);
    void      GetIndexRange(RainbowChain * pChain, uint64_t pChainCount, uint64_t nChainIndex, uint64_t & nChainIndexFrom, uint64_t & nChainIndexTo);

private:
    bool      CheckAlarm(RainbowChain *pChain, uint64_t nGuessedPos, uint64_t testV);
    void      SearchTableChunk(RainbowChain * pChain,int pChainCount);
    void      SearchRainbowTable(const char * fileName);
    void      InitEndKeys(uint64_t key);

private:
    RainbowChainWalk  m_cwc;
    RainbowCipherSet *p_cs;

private:
    /// Statistics Private Data
    struct timeval   m_diskTime, m_totalTime, m_initTime, m_compareTime;
    uint64_t         m_totalChains;
    uint64_t         m_falseAlarms;

private:
    static MemoryPool mp;
    vector<uint64_t> pEndKeys;
    vector<uint64_t> pVerified;
};

};

#endif
