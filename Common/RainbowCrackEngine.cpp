// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "TimeStamp.h"
using namespace utils;

#include "RainbowCrackEngine.h"

namespace rainbowcrack
{

MemoryPool RainbowCrackEngine::mp;

RainbowCrackEngine::RainbowCrackEngine() : m_totalChains(0), m_falseAlarms(0)
{
    m_diskTime.tv_sec   = 0; m_diskTime.tv_usec  = 0;
    m_totalTime.tv_sec  = 0; m_totalTime.tv_usec = 0;
    m_initTime.tv_sec   = 0; m_initTime.tv_usec  = 0;
    m_compareTime.tv_sec = 0; m_compareTime.tv_usec = 0;
    
    p_cs = RainbowCipherSet::GetInstance();
}

uint64_t RainbowCrackEngine::BinarySearch(RainbowChain * pChain, uint64_t pChainCount, uint64_t nIndex)
{
    uint64_t low = 0, high = pChainCount;

    while(low < high)
    {
        uint64_t mid = (low+high)/2;

        if(pChain[mid].nEndKey == nIndex) return mid;
        else if(pChain[mid].nEndKey < nIndex) low = mid + 1;
        else high = mid;
    }

    return low;
}

void RainbowCrackEngine::GetIndexRange(RainbowChain * pChain,uint64_t pChainCount, uint64_t nChainIndex, uint64_t &nChainIndexFrom, uint64_t &nChainIndexTo)
{
    nChainIndexFrom = nChainIndex;
    nChainIndexTo   = nChainIndex;

    while(nChainIndexFrom > 0)
    {
        if(pChain[nChainIndexFrom - 1].nEndKey == pChain[nChainIndex].nEndKey)
            nChainIndexFrom--;
        else
            break;
    }

    while(nChainIndexTo < pChainCount)
    {
        if(pChain[nChainIndexTo + 1].nEndKey == pChain[nChainIndex].nEndKey)
            nChainIndexTo++;
        else break;
    }
}

bool RainbowCrackEngine::CheckAlarm(RainbowChain *pChain, uint64_t nGuessPos, uint64_t testV)
{
    RainbowChainWalk cwc;

    uint64_t nPos = 0, old = pChain -> nStartKey;

    cwc.SetKey(pChain -> nStartKey);

    for(; nPos <= nGuessPos; nPos++)
    {
        old = cwc.GetKey(); cwc.KeyToCipher(); cwc.KeyReduction(nPos);
    }

    if(cwc.GetKey() == pVerified[nGuessPos])
    {
        cout << "plaintext of " << cwc.GetKey() << " is " << old << endl;
        p_cs -> AddResult(p_cs -> GetLastKey(), old);

        return true;
    }

    return false;
}

struct timeval RainbowCrackEngine::GetTotalTime()
{
   return m_totalTime;
}

struct timeval RainbowCrackEngine::GetInitTime()
{
   return m_initTime;
}

struct timeval RainbowCrackEngine::GetDiskTime()
{
    return m_diskTime;
}

struct timeval RainbowCrackEngine::GetCompareTime()
{
    return m_compareTime;
}

uint64_t RainbowCrackEngine::GetTotalChains()
{
    return m_totalChains;
}

uint64_t RainbowCrackEngine::GetFalseAlarms()
{
    return m_falseAlarms;
}

void RainbowCrackEngine::InitEndKeys(uint64_t key)
{
    pEndKeys  = vector<uint64_t>(RainbowChainWalk::m_chainLen, 0);
    pVerified = vector<uint64_t>(RainbowChainWalk::m_chainLen, 0);

    for(uint32_t nGuessPos = 0; nGuessPos < RainbowChainWalk::m_chainLen; nGuessPos++)
    {
        m_cwc.SetKey(key);
        m_cwc.KeyReduction(nGuessPos);

        pVerified[nGuessPos] = m_cwc.GetKey();

        for(uint32_t nIndex = nGuessPos + 1; nIndex < RainbowChainWalk::m_chainLen; nIndex++)
        {
            m_cwc.KeyToCipher();
            m_cwc.KeyReduction(nIndex);
        }

        pEndKeys[nGuessPos] = m_cwc.GetKey();
    }
}

void RainbowCrackEngine::SearchRainbowTable(const char *fileName)
{
    char str[256]; FILE *file; RainbowChain *pChain;
    uint64_t fileLen, nAllocateSize, nDataRead;
    
    if((file = fopen(fileName, "rb")) == NULL)
    {
        printf("SearchRainbowTable: fopen error\n");
        return;
    }

    fileLen = GetFileLen(file);

    assert(fileLen % 16 == 0);

    cout << RainbowChainWalk::m_chainCount << " " << fileLen << endl;

    if(fileLen % 16 != 0 || RainbowChainWalk::m_chainCount*16 != fileLen)
    {
        printf("file length check error\n");
        return;
    }

    if((pChain = (RainbowChain*)mp.Allocate(fileLen, nAllocateSize)) == NULL)
    {
        printf("SearchRainbowTable: allocate error\n");
        return;
    }

    nAllocateSize = nAllocateSize / 16 * 16;
    fseek(file, 0, SEEK_SET);

    while(true)
    {
        if(fileLen == (uint64_t)ftell(file)) break;

        TimeStamp tmps; tmps.StartTime();

        nDataRead = fread(pChain, 1, nAllocateSize, file);

        if(nDataRead != nAllocateSize)
            cout << "Warning nDataRead: " << nDataRead << ", nAllocateSize: "<< nAllocateSize << "\n";
        sprintf(str, "%lld bytes read, disk access time: ", (long long)nAllocateSize);

        tmps.StopTime(str); tmps.AddTime(m_totalTime); tmps.AddTime(m_diskTime);

        tmps.StartTime();

        SearchTableChunk(pChain, nDataRead >> 4);

        sprintf(str, "cryptanalysis time: ");

        tmps.StopTime(str); tmps.AddTime(m_totalTime); tmps.AddTime(m_compareTime);
    }

    p_cs -> Done();

    fclose(file);
}

void RainbowCrackEngine::SearchTableChunk(RainbowChain *pChain, int pChainCount)
{
    uint64_t nFalseAlarm = 0, nIndex, nGuessPos = 0;
    uint64_t key = p_cs -> GetLastKey();

    cout << "Searching for key: " << key << "...\n";

    for(;nGuessPos < RainbowChainWalk::m_chainLen; nGuessPos++)
    {
        uint64_t nMathingIndexE = BinarySearch(pChain, pChainCount, pEndKeys[nGuessPos]);

        if(pChain[nMathingIndexE].nEndKey == pEndKeys[nGuessPos])
        {
            uint64_t nMathingIndexEFrom, nMathingIndexETo;
            GetIndexRange(pChain, pChainCount, nMathingIndexE,nMathingIndexEFrom,nMathingIndexETo);

            for(nIndex = nMathingIndexEFrom; nIndex <= nMathingIndexETo; nIndex++)
            {
                if(CheckAlarm(pChain + nIndex, nGuessPos, pEndKeys[nGuessPos]))
                { } //goto NEXT_HASH;
                else nFalseAlarm++;
            }
        }

        if(nGuessPos % 1000 == 0) cout << "nGuessPos " << nGuessPos << endl;
    }
    m_totalChains += pChainCount; m_falseAlarms += nFalseAlarm;
}

void RainbowCrackEngine::Run(const char *fileName, const char *type)
{
    uint64_t nChainLen, nChainCount;

    if(AnylysisFileName(fileName, nChainLen, nChainCount) == false)
    {
        printf("fileName format error\n");
        return;
    }

    cout << "\nnChainLen: " << nChainLen << ", nChainCount: " << nChainCount << "\n";
    RainbowChainWalk::SetChainInfo(nChainLen, nChainCount, type);

    int index = 0;

    while(p_cs -> Finished())
    {
        cout << "-------------------------------------------------------" << endl;
        cout << "Time: " << index++ << " key: " << p_cs -> GetLastKey() << "\n\n";

        TimeStamp tmps; tmps.StartTime();

        InitEndKeys(p_cs -> GetLastKey());

        tmps.StopTime("Init Time: "); 
        tmps.AddTime(m_totalTime); tmps.AddTime(m_initTime);

        SearchRainbowTable(fileName);

        cout << "-------------------------------------------------------" << endl;
    }
}

};
