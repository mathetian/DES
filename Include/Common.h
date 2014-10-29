// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef _Common_H
#define _Common_H

#include <map>
#include <queue>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
using namespace std;

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <openssl/des.h>
#include <openssl/rand.h>

#ifdef _WIN32
#pragma warning(disable : 4786)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)
#include <Windows.h>
#else
#include <sys/sysinfo.h>
#include <sys/time.h>
#endif

#define CHAIN_IN_MEMORY_MAX 1024

namespace utils
{

class RainbowChain
{
public:
    uint64_t nStartKey, nEndKey;
    bool operator < (const RainbowChain &m) const
    {
        return nEndKey < m.nEndKey;
    }
};

extern uint64_t GetFileLen(FILE*file);

extern void Logo();

extern uint64_t GetAvailPhysMemorySize();

extern bool AnylysisFileName(const string &filename, uint64_t & chainLen, uint64_t & chainCount);

#ifdef _WIN32
inline uint64_t atoll(const char * str)
{
    uint64_t rs;
    istringstream ist(str);
    ist >> rs;

    return rs;
}
#endif

#define fseek64 fseek
#define ftell64 ftell

};

#endif