// Copyright (c) 2014 The RainbowCrack Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <openssl/des.h>

#include "Common.h"
#include "TimeStamp.h"
using namespace utils;

void Test_DES(unsigned char* pPlain, int nPlainLen, unsigned char* pHash)
{
	static unsigned char m_dplainText[8] = {0x6D,0x6F,0x29,0x5A,0x30,0x55,0x32,0x28};

    des_key_schedule ks;
    des_cblock key_56;
    memcpy(key_56, pPlain, nPlainLen);
    DES_set_key_unchecked(&key_56, &ks);
    unsigned char out[8];
    des_ecb_encrypt(&m_dplainText, &out, ks, DES_ENCRYPT);
    memcpy(pHash, out, 8);
}

void Test_DES_Encrypt(const char *filename)
{
	FILE *file = fopen(filename, "rb"); assert(file);
	RainbowChain     chain;
		
	while(fread((char*)&chain, sizeof(RainbowChain), 1, file) == 1)
	{
		unsigned char result[8];
		Test_DES((unsigned char*)&chain.nStartKey, 8, result);
		uint64_t u_val = *(uint64_t*)result;
		assert(u_val == chain.nEndKey);
	}

	fclose(file);
}

int main()
{
	Test_DES_Encrypt("TestCaseGenerator.txt");
	cout << "Passed All Tests" << endl;
	
	return 0;
}