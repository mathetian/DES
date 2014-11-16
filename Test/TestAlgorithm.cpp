// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "RainbowAlgorithm.h"

void Usage()
{
    Logo();
    printf("Usage: test type filename\n");

    printf("example 1: test des/md5/sha1/hmac filename\n");
}

typedef void (*HASHROUTINE)(unsigned char *pPlain, int nPlainLen, unsigned char *pHash);

/// Test Algorithm
void DoTest_1(const char *type, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    assert(file);
    RainbowChain chain;

    HASHROUTINE algorithm;
    if(strcmp(type, "des") == 0) algorithm = HASH_DES;
    else if(strcmp(type, "md5") == 0) algorithm = HASH_MD5;
    else if(strcmp(type, "sha1") == 0) algorithm = HASH_SHA1;
    else if(strcmp(type, "sha1hmac") == 0) algorithm = HASH_SHA1_HMAC;
    else if(strcmp(type, "md5hmac") == 0) algorithm = HASH_MD5_HMAC;
    else assert(0);

    while(fread((char*)&chain, sizeof(RainbowChain), 1, file) == 1)
    {
        unsigned char result[8];
        algorithm((unsigned char*)&chain.nStartKey, 8, result);
        uint64_t u_val = *(uint64_t*)result;
        assert(u_val == chain.nEndKey);
    }

    fclose(file);
}

/// Test CPU & GPU
void DoTest_2()
{

}

/// Test Sort
void DoTest_3()
{

}

/// Test Performance
void DoTest_4()
{

}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        Usage();
        return 0;
    }
    DoTest_1(argv[1], argv[2]);
    cout << "Passed All Tests" << endl;

    return 0;
}