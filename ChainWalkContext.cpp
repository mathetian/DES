#include "ChainWalkContext.h"

uint64_t   ChainWalkContext::m_plainText     = 0x305532286D6F295A;
/*uint64_t   ChainWalkContext::m_keySpaceTotal = (1ull << 56) - 1;*/
uint64_t   ChainWalkContext::m_keySpaceTotal = (1ull << 20) - 1;
uint64_t   ChainWalkContext::m_keySpaceTotalT = (1ull << 24) - (1ull << 8) - 2 - (1ull << 16) - (1ull << 17);

uint64_t   ChainWalkContext::m_chainLen;
uint64_t   ChainWalkContext::m_chainCount;
unsigned char ChainWalkContext::m_dplainText[8] = {0x6B,0x05,0x6E,0x18,0x75,0x9F,0x5C,0xCA};

ChainWalkContext::ChainWalkContext()
{
}

ChainWalkContext::~ChainWalkContext()
{
}

void ChainWalkContext::SetChainInfo(uint64_t chainLen, uint64_t chainCount)
{
	m_chainLen   = chainLen;
	m_chainCount = chainCount;
}

uint64_t ChainWalkContext::GetRandomKey()
{
	RAND_bytes((unsigned char*)&m_nIndex,8);
	m_nIndex = m_nIndex & m_keySpaceTotalT;
	return m_nIndex;
}

/**
    des_cblock: typedef unsigned char DES_cblock[8];
**/
/**
typedef struct DES_ks
{
    union
	{
		DES_cblock cblock;
		DES_LONG deslong[2];
	} ks[16];
} DES_key_schedule;
DES_LONG is 'unsigned int'
**/

void ChainWalkContext::KeyToCipher()
{
	des_key_schedule ks;unsigned char out[8];
	SetupDESKey(m_nIndex,ks); memset(out,0,8);
	des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);
	CipherToKey(out);
}

void ChainWalkContext::CipherToKey(unsigned char * out)
{
	Arr7ToU56(out, m_nIndex);
	m_nIndex &= m_keySpaceTotalT;
}

/**
	Still exist the same problem
**/
void ChainWalkContext::KeyReduction(int nPos)
{
	/**
		Exist very big problem, will worse the distribution.
	**/
	m_nIndex = (m_nIndex + nPos) & m_keySpaceTotalT;	
}

uint64_t ChainWalkContext::GetKey()
{
	return m_nIndex & m_keySpaceTotalT;
}

void 	 ChainWalkContext::SetKey(uint64_t key)
{
	m_nIndex = key & m_keySpaceTotalT;
}

uint64_t ChainWalkContext::Crypt(uint64_t key)
{
	des_key_schedule ks;unsigned char out[8];
	SetupDESKey(key,ks); memset(out,0,8);
	des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);
	Arr7ToU56(out, key); key &= m_keySpaceTotal;
	return key;
}