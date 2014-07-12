// Copyright (c) 2014 The DESCrack Authors. All rights reserved.
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

namespace utils
{

class RainbowChain
{
public:
    uint64_t nStartKey, nEndKey;
    bool operator < (const RainbowChain &m) const;
};

extern uint64_t GetFileLen(FILE*file);

extern void Logo();

extern uint64_t GetAvailPhysMemorySize();

extern void U56ToArr7(const uint64_t & key56, unsigned char * key_56);


extern void Arr7ToU56(const unsigned char * key_56, uint64_t & key56);
/**
	des_cblock: typedef unsigned char DES_cblock[8]
**/

extern void SetupDESKey(const uint64_t&key56, des_key_schedule &ks);

extern bool AnylysisFileName(const char * filename, uint64_t & chainLen, uint64_t & chainCount);

#define CHAIN_IN_MEMORY_MAX 1024

#ifdef _WIN32
inline uint64_t atoll(const char * str)
{
    uint64_t rs;
    istringstream ist(str);
    ist >> rs;

    return rs;
}

extern string GetLastErrorStdStr();
#endif

#define _fseeki64 fseek
#define _ftelli64 ftell

};

#endif