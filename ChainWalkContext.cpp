#include "ChainWalkContext.h"

uint64_t   ChainWalkContext::m_plainText     = 0x305532286D6F295A;
uint64_t   ChainWalkContext::m_keySpaceTotal = (1ull << 56) - 1;
int        ChainWalkContext::m_chainLen;
int        ChainWalkContext::m_chainCount;
des_cblock ChainWalkContext::m_dplainText    = {0x30,0x55,0x32,0x28,0x6D,0x6F,0x29,0x5A};

ChainWalkContext::ChainWalkContext()
{
}

ChainWalkContext::~ChainWalkContext()
{
}

void ChainWalkContext::SetChainInfo(int chainLen, int chainCount)
{
	m_chainLen   = chainLen;
	m_chainCount = chainCount;
}

uint64_t ChainWalkContext::GetRandomKey()
{
	RAND_bytes((unsigned char*)&m_nIndex,8);
	m_nIndex = m_nIndex & m_keySpaceTotal;
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

void ChainWalkContext::KeyToHash()
{
	des_key_schedule ks;unsigned char out[8];
	SetupDESKey(m_nIndex,ks); memset(out,0,8);
	des_ecb_encrypt(&m_dplainText,&out,ks,DES_ENCRYPT);
	Arr7ToU56(out,m_nIndex);
}

void ChainWalkContext::HashToKey(int nPos)
{
	m_nIndex = (m_nIndex + nPos) & m_keySpaceTotal;
}

uint64_t ChainWalkContext::GetKey()
{
	return m_nIndex;
}
