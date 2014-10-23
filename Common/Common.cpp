// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"

namespace utils
{

void Logo()
{
    printf("RainbowCrack 1.0\n 	Make an implementation of Time-and-Memory Tradeoff Technology\n 	By Tian Yulong(mathetian@gmail.com)\n\n");
}

uint64_t GetFileLen(FILE* file)
{
    uint64_t pos = ftell(file);
    fseek(file, 0, SEEK_END);
    uint64_t len = ftell(file);

    fseek(file, pos, SEEK_SET);

    return len;
}

uint64_t GetAvailPhysMemorySize()
{
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof (statex);
    GlobalMemoryStatusEx (&statex);
    return statex.ullAvailPhys;
#else
    struct sysinfo info;
    sysinfo(&info);
    return info.freeram;
#endif
}

/**
** DES
**/
void U56ToArr7(const uint64_t & key56, unsigned char * key_56)
{
    key_56 = (unsigned char *)&key_56;
}

void Arr7ToU56(const unsigned char * key_56, uint64_t & key56)
{
    key56 = *(uint64_t*)key_56;
}

void SetupDESKey(const uint64_t & key56,des_key_schedule & ks)
{
    des_cblock key_56;

    U56ToArr7(key56, key_56);

    DES_set_key_unchecked(&key_56, &ks);
}

bool AnylysisFileName(const char * filename, uint64_t & chainLen, uint64_t & chainCount)
{
    int len = strlen(filename), i = 0, j;
    if(len <= 6 || filename[3] != '_') return false;
    char str[256];
    memset(str, 0, sizeof(str));
    for(i = 4; i< len; i++) if(filename[i] == '-') break;
    if(i == len || i == 3) return false;
    memcpy(str,filename + 4, i - 4);

    chainLen = atoll(str);

    memset(str, 0, sizeof(str));
    for(j = i + 1; j < len; j++) if(filename[j] == '_') break;
    if(j == len || j == i+1) return false;
    memcpy(str,filename + i + 1,j - i - 1);

    chainCount = atoll(str);

    return true;
}

};