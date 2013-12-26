#ifndef _CHAIN_WALK_CONTEXT_H
#define _CHAIN_WALK_CONTEXT_H

#include <stdlib.h>
#include <stdint.h>
#include <openssl/rand.h>
#include <openssl/des.h>

#define HASH_LEN 8

class ChainWalkContext{
public:
	ChainWalkContext();
	virtual ~ ChainWalkContext();
private:
	static uint64_t m_plainText;
	static uint64_t m_keySpaceTotal;	
	static int 		m_chainLen;
	static int 		m_chainCount;
	static des_cblock m_dplainText;
	
public:
	static void 	SetChainInfo(int chainlen,int chainCount);

public:
	void 	 		KeyToHash();
	void 	 		HashToKey(int nPos);
	uint64_t 		GetRandomKey();
	uint64_t 		GetKey();

private:
	uint64_t m_nIndex;
};

#endif