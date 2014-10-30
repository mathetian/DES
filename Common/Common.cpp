// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "Common.h"

namespace utils
{

void Logo()
{
    cout << "RainbowCrack 1.0" << endl;
    cout << "Make an implementation of Time-and-Memory Tradeoff Technology" << endl;
    cout << "     By Tian Yulong (yulong.ti@gmail.com)\n\n" << endl;
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

bool AnylysisFileName(const string &filename, uint64_t &chainLen, uint64_t &chainCount)
{
    int len = filename.size(), i = 0, j;
    bool flag = true;
    do
    {
        while(i < len && filename[i] != '_') i++;
        i++;
        if(i >= len)
        {
            flag = false;
            continue;
        }

        j = i + 1;
        while(j < len && filename[j] != '-') j++;
        if(j + 2 >= len)
        {
            flag = false;
            continue;
        }
        chainLen = atoi(filename.substr(i, j - i).c_str());

        j++;
        i = j;
        while(j < len && filename[j] != '_') j++;
        if(j >= len)
        {
            flag = false;
            continue;
        }
        chainCount = atoi(filename.substr(i, j - i).c_str());
    }
    while(0);

    cout << flag << " " << chainLen << " " << chainCount << endl;
    return flag;
}

};