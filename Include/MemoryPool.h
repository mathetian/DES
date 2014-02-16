#ifndef _MEMORY_POOL_H
#define _MEMORY_POOL_H
#include "Common.h"

class MemoryPool{
public:
	MemoryPool();
	virtual ~ MemoryPool();

public:
	unsigned char*Allocate(uint64_t nFileLen,uint64_t&nAllocateSize);

private:
	unsigned char      * m_pMem;
	uint64_t    m_nMemSize;
	uint64_t    m_nMemMax;
	uint64_t    m_nAvailPhys;
};
#endif