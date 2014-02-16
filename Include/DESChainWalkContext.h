#ifndef _CHAIN_WALK_CONTEXT_H
#define _CHAIN_WALK_CONTEXT_H

#include <openssl/rand.h>
#include <string>
using namespace std;

#include "DESCommon.h"

#define HASH_LEN 8

class DESChainWalkContext
{
public:
    DESChainWalkContext();
    virtual ~ DESChainWalkContext();

private:
    static uint64_t   m_plainText;
    static uint64_t   m_keySpaceTotal;
    static unsigned char m_dplainText[8];


public:
    static uint64_t   m_chainLen;
    static uint64_t   m_chainCount;
    static uint64_t   m_keySpaceTotalT;

public:
    static void 	SetChainInfo(uint64_t chainLen,uint64_t chainCount);

public:
    void 	 		KeyToCipher();
    void 	 		KeyReduction(int nPos);
    uint64_t 		GetRandomKey();
    uint64_t 		GetKey();
    void 			SetKey(uint64_t m_nIndex);

    uint64_t 		Crypt(uint64_t key);

private:
    void 			CipherToKey(unsigned char * out);

private:
    uint64_t m_nIndex;
};

#endif