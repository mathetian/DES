#ifndef _CHAIN_WALK_CONTEXT_H
#define _CHAIN_WALK_CONTEXT_H


#include <openssl/rand.h>
#include <stdint.h>
#include <string>
using namespace std;

#include "Common.h"

#define HASH_LEN 8

class ChainWalkContext{
public:
	ChainWalkContext();
	virtual ~ ChainWalkContext();
private:
	static uint64_t   m_plainText;
	static uint64_t   m_keySpaceTotal;	
	static des_cblock m_dplainText;
public:
	static int 		  m_chainLen;
	static int 		  m_chainCount;
public:
	static void 	SetChainInfo(int chainLen,int chainCount);
	static void 	SetupWithPathName(const string & fileName);
public:
	void 	 		KeyToHash();
	void 	 		HashToKey(int nPos);
	uint64_t 		GetRandomKey();
	uint64_t 		GetKey();
	void 			SetKey(uint64_t m_nIndex);
private:
	uint64_t m_nIndex;
};

#endif